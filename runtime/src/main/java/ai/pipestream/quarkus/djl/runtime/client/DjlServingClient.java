package ai.pipestream.quarkus.djl.runtime.client;

import io.smallrye.mutiny.Uni;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.PathParam;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.QueryParam;
import jakarta.ws.rs.core.MediaType;
import org.eclipse.microprofile.rest.client.inject.RegisterRestClient;

/**
 * MicroProfile REST client for DJL Serving's HTTP API.
 *
 * <p>Wired via {@code quarkus.rest-client.djl-serving.url} at runtime.
 * DJL Serving exposes inference on {@code POST /predictions/{modelName}}
 * and management on {@code GET/POST /models}. This interface covers the
 * subset used by {@link ai.pipestream.quarkus.djl.runtime.DjlModelRegistry}
 * and {@link ai.pipestream.quarkus.djl.runtime.DjlServingBackend}.
 */
@Path("/")
@RegisterRestClient(configKey = "djl-serving")
public interface DjlServingClient {

    @GET
    @Path("/ping")
    Uni<String> ping();

    @POST
    @Path("/predictions/{modelName}")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    Uni<JsonArray> predict(@PathParam("modelName") String modelName, JsonObject input);

    @GET
    @Path("/models")
    @Produces(MediaType.APPLICATION_JSON)
    Uni<JsonObject> listModels();

    /**
     * Registers (loads) a model into DJL Serving via its management API.
     * Example: {@code POST /models?url=djl://...&model_name=minilm&engine=PyTorch}
     *
     * @param url       model URL (e.g. {@code djl://ai.djl.huggingface.pytorch/sentence-transformers/all-MiniLM-L6-v2})
     * @param modelName name to register the model under
     * @param engine    inference engine (e.g. {@code PyTorch})
     */
    @POST
    @Path("/models")
    @Produces(MediaType.APPLICATION_JSON)
    Uni<JsonObject> registerModel(
            @QueryParam("url") String url,
            @QueryParam("model_name") String modelName,
            @QueryParam("engine") String engine);
}
