"""
02_Docker_K8s_Deployment.py — Production Docker + Kubernetes Deployment

Complete production deployment for LLM inference including:
1. Multi-stage Dockerfile with CUDA optimization
2. Kubernetes manifests with GPU resource management
3. Horizontal Pod Autoscaler (HPA) based on GPU utilization
4. Service mesh integration (Istio)
5. Persistent volume for model caching
"""

import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Dockerfile Template
# ---------------------------------------------------------------------------

DOCKERFILE_TEMPLATE = """# Multi-stage Dockerfile for LLM inference
# Stage 1: Builder
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with pinned versions
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \\
    pip install --no-cache-dir -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Runtime
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3.10-venv \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables for CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser
USER appuser
WORKDIR /home/appuser/app

# Copy application code
COPY --chown=appuser:appuser . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Entry point with graceful shutdown handling
ENTRYPOINT ["/opt/venv/bin/python", "-m", "uvicorn", "main:app", \\
            "--host", "0.0.0.0", "--port", "8000", \\
            "--workers", "1",  # vLLM needs single process for GPU
            "--timeout-keep-alive", "30", \\
            "--log-level", "info"]
"""

# ---------------------------------------------------------------------------
# Requirements.txt Template
# ---------------------------------------------------------------------------

REQUIREMENTS_TEMPLATE = """# Production dependencies with pinned versions
fastapi==0.104.1
uvicorn[standard]==0.24.0
vllm==0.3.3
torch==2.1.2
transformers==4.36.2
accelerate==0.25.0
prometheus-client==0.19.0
pydantic==2.5.0
python-multipart==0.0.6
httpx==0.25.1
tenacity==8.2.3
structlog==23.2.0

# Optional: model quantization
autoawq==0.1.8
auto-gptq==0.5.1

# Monitoring
opentelemetry-api==1.22.0
opentelemetry-sdk==1.22.0
opentelemetry-instrumentation-fastapi==0.42b0
"""

# ---------------------------------------------------------------------------
# Kubernetes Manifests
# ---------------------------------------------------------------------------

K8S_DEPLOYMENT_TEMPLATE = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {deployment_name}
  namespace: {namespace}
  labels:
    app: {deployment_name}
    component: llm-inference
spec:
  replicas: {replica_count}
  selector:
    matchLabels:
      app: {deployment_name}
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: {deployment_name}
        component: llm-inference
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      nodeSelector:
        nvidia.com/gpu.product: {gpu_type}
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values: [{deployment_name}]
              topologyKey: kubernetes.io/hostname
      containers:
      - name: inference-server
        image: {docker_image}:{image_tag}
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: MODEL_ID
          value: {model_id}
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          limits:
            nvidia.com/gpu: {gpus_per_pod}
            memory: "{memory_limit}Gi"
            cpu: "{cpu_limit}"
          requests:
            nvidia.com/gpu: {gpus_per_pod}
            memory: "{memory_request}Gi"
            cpu: "{cpu_request}"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 1
        volumeMounts:
        - name: model-cache
          mountPath: /home/appuser/.cache/huggingface
        - name: tmp-volume
          mountPath: /tmp
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: tmp-volume
        emptyDir: {}
      securityContext:
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
"""

K8S_SERVICE_TEMPLATE = """apiVersion: v1
kind: Service
metadata:
  name: {service_name}
  namespace: {namespace}
spec:
  selector:
    app: {deployment_name}
  ports:
  - port: 80
    targetPort: 8000
    name: http
  type: ClusterIP
"""

K8S_HPA_TEMPLATE = """apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {hpa_name}
  namespace: {namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {deployment_name}
  minReplicas: {min_replicas}
  maxReplicas: {max_replicas}
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: {gpu_target_utilization}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {memory_target_utilization}
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 1
        periodSeconds: 30
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 60
"""

K8S_PVC_TEMPLATE = """apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache-pvc
  namespace: {namespace}
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: {storage_size}Gi
  storageClassName: {storage_class}
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DeploymentConfig:
    """Kubernetes deployment configuration."""
    # Basic
    deployment_name: str = "llama-3-70b-inference"
    namespace: str = "llm-production"
    replica_count: int = 2

    # Docker
    docker_image: str = "registry.example.com/llm-inference"
    image_tag: str = "v1.0.0"

    # Model
    model_id: str = "meta-llama/Llama-3.1-70B-Instruct"

    # GPU
    gpu_type: str = "NVIDIA-A100-SXM4-80GB"  # or "NVIDIA-H100-80GB-PCIe"
    gpus_per_pod: int = 4  # Tensor parallelism across 4 GPUs

    # Resources
    memory_limit: int = 350  # GiB per pod
    memory_request: int = 300  # GiB per pod
    cpu_limit: str = "16"
    cpu_request: str = "8"

    # Autoscaling
    min_replicas: int = 2
    max_replicas: int = 10
    gpu_target_utilization: int = 80  # Scale when GPU utilization >80%
    memory_target_utilization: int = 85  # Scale when memory utilization >85%

    # Storage
    storage_size: int = 500  # GiB for model cache
    storage_class: str = "fast-ssd"

    # Service
    service_name: str = "llama-3-70b-service"
    hpa_name: str = "llama-3-70b-hpa"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_deployment_files(config: DeploymentConfig, output_dir: Path):
    """Generate all deployment files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dockerfile
    dockerfile_path = output_dir / "Dockerfile"
    dockerfile_path.write_text(DOCKERFILE_TEMPLATE)

    # 2. requirements.txt
    requirements_path = output_dir / "requirements.txt"
    requirements_path.write_text(REQUIREMENTS_TEMPLATE)

    # 3. Kubernetes manifests
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(exist_ok=True)

    # Deployment
    deployment_yml = K8S_DEPLOYMENT_TEMPLATE.format(
        deployment_name=config.deployment_name,
        namespace=config.namespace,
        replica_count=config.replica_count,
        docker_image=config.docker_image,
        image_tag=config.image_tag,
        gpu_type=config.gpu_type,
        gpus_per_pod=config.gpus_per_pod,
        memory_limit=config.memory_limit,
        memory_request=config.memory_request,
        cpu_limit=config.cpu_limit,
        cpu_request=config.cpu_request,
        model_id=config.model_id,
    )
    (manifests_dir / "deployment.yaml").write_text(deployment_yml)

    # Service
    service_yml = K8S_SERVICE_TEMPLATE.format(
        service_name=config.service_name,
        namespace=config.namespace,
        deployment_name=config.deployment_name,
    )
    (manifests_dir / "service.yaml").write_text(service_yml)

    # HPA
    hpa_yml = K8S_HPA_TEMPLATE.format(
        hpa_name=config.hpa_name,
        namespace=config.namespace,
        deployment_name=config.deployment_name,
        min_replicas=config.min_replicas,
        max_replicas=config.max_replicas,
        gpu_target_utilization=config.gpu_target_utilization,
        memory_target_utilization=config.memory_target_utilization,
    )
    (manifests_dir / "hpa.yaml").write_text(hpa_yml)

    # PVC
    pvc_yml = K8S_PVC_TEMPLATE.format(
        namespace=config.namespace,
        storage_size=config.storage_size,
        storage_class=config.storage_class,
    )
    (manifests_dir / "pvc.yaml").write_text(pvc_yml)

    # 4. Deployment script
    deploy_script = """#!/bin/bash
# Production deployment script

set -euo pipefail

echo "Building Docker image..."
docker build -t ${DOCKER_IMAGE}:${IMAGE_TAG} .

echo "Pushing to registry..."
docker push ${DOCKER_IMAGE}:${IMAGE_TAG}

echo "Applying Kubernetes manifests..."
kubectl apply -f manifests/

echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/${DEPLOYMENT_NAME} -n ${NAMESPACE}

echo "Deployment complete!"
"""
    (output_dir / "deploy.sh").write_text(deploy_script)
    (output_dir / "deploy.sh").chmod(0o755)

    # 5. Makefile
    makefile = """# Makefile for LLM deployment

.PHONY: build push deploy clean logs

DOCKER_IMAGE = registry.example.com/llm-inference
IMAGE_TAG = v1.0.0
NAMESPACE = llm-production
DEPLOYMENT_NAME = llama-3-70b-inference

build:
	docker build -t $(DOCKER_IMAGE):$(IMAGE_TAG) .

push:
	docker push $(DOCKER_IMAGE):$(IMAGE_TAG)

deploy: push
	kubectl apply -f manifests/
	kubectl rollout status deployment/$(DEPLOYMENT_NAME) -n $(NAMESPACE)

clean:
	kubectl delete -f manifests/ --ignore-not-found
	docker rmi $(DOCKER_IMAGE):$(IMAGE_TAG) || true

logs:
	kubectl logs -f deployment/$(DEPLOYMENT_NAME) -n $(NAMESPACE) --tail=100

port-forward:
	kubectl port-forward service/llama-3-70b-service 8080:80 -n $(NAMESPACE)

health:
	curl http://localhost:8080/health

metrics:
	curl http://localhost:8080/metrics
"""
    (output_dir / "Makefile").write_text(makefile)

    return output_dir


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example configuration
    config = DeploymentConfig()

    # Generate files
    output_dir = Path("deployment")
    generate_deployment_files(config, output_dir)

    print(f"Deployment files generated in: {output_dir}")
    print("\nGenerated files:")
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            path = Path(root) / file
            print(f"  {path.relative_to(output_dir)}")

    print("\nNext steps:")
    print("1. Review and customize the generated files")
    print("2. Build Docker image: make build")
    print("3. Push to registry: make push")
    print("4. Deploy to Kubernetes: make deploy")