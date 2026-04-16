# pipestream-embedder-djl

Quarkus extension implementing the `EmbeddingBackend` SPI over **DJL Serving**
via the KServe v2 gRPC protocol. Drops into any Quarkus application that
depends on `ai.pipestream.module:module-embedder-api` — once on the classpath,
ARC discovers `DjlBackend` automatically through
`@Inject Instance<EmbeddingBackend>`.

## Design

- **Fully reactive.** Every gRPC call returns `Uni<T>`; no blocking stubs, no
  `Uni.createFrom().item(() -> syncBlock)` wrappers. Metadata discovery and
  inference both run on Mutiny end-to-end.
- **Quarkus-native transport.** Uses `@GrpcClient("djl")` so Stork service
  discovery, TLS, interceptors, and deadlines all configure via
  `quarkus.grpc.clients.djl.*` — no hand-rolled `ManagedChannel`.
- **Per-serving-name Uni caching.** First `embed()` call for a given model
  triggers a one-shot `ModelMetadata` RPC; result is memoised with
  `.memoize().indefinitely()`. Subsequent calls skip metadata.

## Configuration

```properties
quarkus.grpc.clients.djl.host=localhost
quarkus.grpc.clients.djl.port=8082

# Service discovery via Stork (Consul, Kubernetes, etc.)
quarkus.grpc.clients.djl.name-resolver=stork
quarkus.stork.djl.service-discovery.type=consul

# TLS
quarkus.grpc.clients.djl.tls-configuration-name=djl-tls
quarkus.tls.djl-tls.trust-store.p12.path=/etc/certs/djl-trust.p12
```

## Modules

- `runtime/` — `DjlBackend`, `DjlModelDescriptor`, `DjlMutinyBatchedClient`,
  KServe v2 proto stubs (generated from the upstream KServe repo at build time)
- `deployment/` — Quarkus processor that registers the backend bean + Jandex
  index over `module-embedder-api`
- `integration-tests/` — live-container IT against a real DJL Serving instance
