package org.hiddenmoon.waterreaction.results;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.json.JSONObject;
import org.junit.Test;

import java.util.List;

public class ResultPayloadTest {
    @Test
    public void payloadMatchesDesktopSchema() throws Exception {
        String text = ResultPayload.create(
                "u-1", "污水", "normal", "initial", 1, 1, 3, "Pixel",
                List.of(new ResultPayload.Tube(1, 2, 3, 4, "已反应", .91234f)));

        JSONObject json = new JSONObject(text);
        assertEquals(1, json.getInt("schema_version"));
        assertEquals("android", json.getString("client_platform"));
        assertEquals("initial", json.getString("app_release_id"));
        assertEquals(3, json.getInt("app_version_code"));
        assertEquals("Pixel", json.getString("device_model"));
        assertTrue(json.getString("captured_at").endsWith("+00:00"));
        assertEquals(.9123, json.getJSONArray("results")
                .getJSONObject(0).getDouble("confidence"), .00001);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsUnsupportedWaterType() {
        ResultPayload.create("u-1", "未知", "normal", "initial", 1, 1, 3, "Pixel", List.of());
    }
}
