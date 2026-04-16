package ai.pipestream.quarkus.djl.runtime;

import ai.pipestream.module.embedder.spi.EmbeddingBackend;
import inference.MutinyGRPCInferenceServiceGrpc;
import io.quarkus.grpc.GrpcClient;
import io.smallrye.mutiny.Uni;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

/**
 * DJL Serving backend. Implements the {@link EmbeddingBackend} SPI over
 * the KServe v2 gRPC protocol using Quarkus's {@link GrpcClient} stack —
 * Stork service discovery, TLS, deadlines, and interceptors all
 * configurable via {@code quarkus.grpc.clients.djl.*} keys.
 *
 * <p><b>Fully reactive.</b> Every gRPC call on the hot path returns a
 * {@link Uni} — no {@code .await()}, no blocking stubs, no
 * {@code Uni.createFrom().item(() -> syncBlock)} wrappers (the older
 * REST-based DJL client had that pattern and it was a footgun: the
 * enclosed code ran on the subscriber thread, which under Vert.x
 * event-loop execution serialised "concurrent" calls into sequential
 * blocking work).
 *
 * <p>Marked {@link Singleton} (not {@code @ApplicationScoped}) so ARC
 * does not generate a client proxy — required because
 * {@link EmbeddingBackend} is discovered via
 * {@code Instance<EmbeddingBackend>} across a Quarkus extension
 * classloader boundary.
 */
@Singleton
public class DjlBackend implements EmbeddingBackend {

    private static final Logger log = LoggerFactory.getLogger(DjlBackend.class);

    @Inject
    @GrpcClient("djl")
    MutinyGRPCInferenceServiceGrpc.MutinyGRPCInferenceServiceStub stub;

    @ConfigProperty(name = "embedder.djl.batch-size", defaultValue = "32")
    int batchSize;

    @ConfigProperty(name = "embedder.djl.timeout-ms", defaultValue = "30000")
    int timeoutMs;

    private final ConcurrentHashMap<String, Uni<DjlMutinyBatchedClient>> clients = new ConcurrentHashMap<>();

    @Override
    public String name() {
        return "djl-serving";
    }

    @Override
    public boolean supports(String servingName) {
        if (servingName == null || servingName.isBlank()) {
            return false;
        }
        clientUni(servingName).subscribe().with(
                c -> {},
                err -> log.debug("DJL probe failed for '{}': {}", servingName, err.getMessage()));
        return true;
    }

    @Override
    public Uni<List<float[]>> embed(String servingName, List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return Uni.createFrom().item(List.of());
        }
        return clientUni(servingName).chain(client -> client.embed(texts));
    }

    private Uni<DjlMutinyBatchedClient> clientUni(String servingName) {
        return clients.computeIfAbsent(servingName, sn -> {
            log.info("Registering DJL Serving client Uni for serving name '{}' (batch={}, timeout={}ms)",
                    sn, batchSize, timeoutMs);
            return DjlMutinyBatchedClient.create(stub, sn, batchSize, timeoutMs)
                    .memoize().indefinitely();
        });
    }
}
