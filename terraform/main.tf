# Provider
provider "kubernetes" {
	config_context = "minikube"
}

resource "kubernetes_namespace" "example" {
	metadata {
		name = "microservices"
	}
}
