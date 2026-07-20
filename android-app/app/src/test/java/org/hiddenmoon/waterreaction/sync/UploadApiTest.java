package org.hiddenmoon.waterreaction.sync;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;

import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.nio.file.Files;

public class UploadApiTest {
    @Rule public final TemporaryFolder temporaryFolder = new TemporaryFolder();
    private MockWebServer server;

    @Before public void setUp() throws Exception {
        server = new MockWebServer();
        server.start();
    }

    @After public void tearDown() throws Exception {
        server.shutdown();
    }

    @Test
    public void registersAndroidAndUploadsDesktopCompatibleMultipart() throws Exception {
        server.enqueue(json(201, "{\"installation_id\":\"i-1\",\"token\":\"t-1\"}"));
        server.enqueue(json(201, "{\"status\":\"created\"}"));
        MemoryCredentials credentials = new MemoryCredentials();
        UploadApi api = new UploadApi(config(), credentials, null);
        File archive = temporaryFolder.newFile("u-1.zip");
        Files.write(archive.toPath(), new byte[]{1, 2, 3});

        UploadApi.Response result = api.upload(archive);

        assertEquals(201, result.statusCode);
        RecordedRequest registration = server.takeRequest();
        String registrationBody = registration.getBody().readString(UTF_8);
        assertEquals("/api/v1/client/register", registration.getPath());
        assertTrue(registrationBody.contains("\"client_platform\":\"android\""));
        assertTrue(registrationBody.contains("\"bootstrap_token\":\"bootstrap\""));
        RecordedRequest upload = server.takeRequest();
        assertEquals("Bearer t-1", upload.getHeader("Authorization"));
        assertEquals("i-1", upload.getHeader("X-Installation-ID"));
        assertTrue(upload.getBody().readString(UTF_8).contains("name=\"file\""));
    }

    @Test
    public void invalidCredentialsReregisterOnlyOnce() throws Exception {
        MemoryCredentials credentials = new MemoryCredentials();
        credentials.save(new UploadApi.Credentials("old-i", "old-t"));
        server.enqueue(json(401, "{\"code\":\"invalid_credentials\"}"));
        server.enqueue(json(201, "{\"installation_id\":\"new-i\",\"token\":\"new-t\"}"));
        server.enqueue(json(208, "{\"status\":\"already_received\"}"));
        File archive = temporaryFolder.newFile("u-2.zip");
        Files.write(archive.toPath(), new byte[]{7});

        UploadApi.Response result = new UploadApi(config(), credentials, null).upload(archive);

        assertEquals(208, result.statusCode);
        assertEquals(3, server.getRequestCount());
        assertEquals("new-i", credentials.load().installationId);
    }

    private ServerConfig config() {
        String base = server.url("/").toString();
        return new ServerConfig(base.substring(0, base.length() - 1), "bootstrap", "initial", 1, 1);
    }

    private static MockResponse json(int code, String body) {
        return new MockResponse().setResponseCode(code)
                .setHeader("Content-Type", "application/json")
                .setBody(body);
    }

    private static final class MemoryCredentials implements UploadApi.CredentialStore {
        private UploadApi.Credentials value;
        @Override public UploadApi.Credentials load() { return value; }
        @Override public void save(UploadApi.Credentials credentials) { value = credentials; }
        @Override public void clear() { value = null; }
    }
}
