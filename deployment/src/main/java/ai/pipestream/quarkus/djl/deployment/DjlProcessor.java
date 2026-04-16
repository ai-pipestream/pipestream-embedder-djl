package ai.pipestream.quarkus.djl.deployment;

import ai.pipestream.quarkus.djl.runtime.DjlBackend;
import io.quarkus.arc.deployment.AdditionalBeanBuildItem;
import io.quarkus.deployment.annotations.BuildStep;
import io.quarkus.deployment.builditem.FeatureBuildItem;
import io.quarkus.deployment.builditem.IndexDependencyBuildItem;

/**
 * Build-time configuration for the DJL Serving Embeddings extension.
 *
 * <p>Registers {@link DjlBackend} as an ARC-managed {@code @Singleton}
 * and indexes the {@code module-embedder-api} jar so ARC's bean-graph
 * builder sees the {@code EmbeddingBackend} interface at build time.
 * The gRPC transport is handled entirely by Quarkus's
 * {@code @GrpcClient} infrastructure — Stork, TLS, interceptors,
 * deadlines all configured via {@code quarkus.grpc.clients.djl.*}
 * config keys.
 */
public class DjlProcessor {

    private static final String FEATURE = "djl-embeddings";

    @BuildStep
    FeatureBuildItem feature() {
        return new FeatureBuildItem(FEATURE);
    }

    @BuildStep
    IndexDependencyBuildItem indexEmbedderApi() {
        return new IndexDependencyBuildItem("ai.pipestream.module", "module-embedder-api");
    }

    @BuildStep
    AdditionalBeanBuildItem beans() {
        return AdditionalBeanBuildItem.builder()
                .addBeanClass(DjlBackend.class)
                .setUnremovable()
                .build();
    }
}
