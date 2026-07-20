package org.hiddenmoon.waterreaction.sync;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class UploadSummaryTest {
    @Test
    public void reportsQueueCountsAndRetryability() {
        UploadSummary summary = new UploadSummary(2, 1, 3, 4, "invalid_payload");

        assertEquals(10, summary.total());
        assertTrue(summary.canRetry());
        assertEquals("invalid_payload", summary.latestError);
    }

    @Test
    public void blockedOnlyQueueDoesNotOfferRetry() {
        assertFalse(new UploadSummary(0, 0, 0, 1, "server_rejected").canRetry());
    }
}
