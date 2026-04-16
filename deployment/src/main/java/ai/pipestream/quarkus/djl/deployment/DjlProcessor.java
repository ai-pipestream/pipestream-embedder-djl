package ai.pipestream.quarkus.djl.deployment;

import ai.pipestream.quarkus.djl.runtime.DjlModelRegistry;
import ai.pipestream.quarkus.djl.runtime.DjlServingBackend;
import ai.pipestream.quarkus.djl.runtime.DjlServingReadinessCheck;
import io.quarkus.arc.deployment.AdditionalBeanBuildItem;
import io.quarkus.deployment.annotations.BuildStep;
import io.quarkus.deployment.builditem.FeatureBuildItem;
import io.quarkus.deployment.builditem.IndexDependencyBuildItem;

/**
 * Build-time configuration for the DJL Serving Embeddings extension.
 *
 * <p>Registers {@link DjlServingBackend}, {@link DjlModelRegistry} and
 * {@link DjlServingReadinessCheck} as ARC beans, and indexes the
 * {@code module-embedder-api} jar so ARC's bean-graph builder can see
 * the {@code EmbeddingBackend} interface at build time.
 *
 * <p>The REST client ({@code DjlServingClient}) is picked up automatically
 * by the {@code quarkus-rest-client} extension via
 * {@code @RegisterRestClient} — no explicit build step needed.
 */
public class DjlProcessor {

    private static final String FEATURE = "djl-serving-embeddings";

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
                .addBeanClass(DjlServingBackend.class)
                .addBeanClass(DjlModelRegistry.class)
                .addBeanClass(DjlServingReadinessCheck.class)
                .setUnremovable()
                .build();
    }
}
