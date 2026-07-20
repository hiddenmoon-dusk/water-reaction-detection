package org.hiddenmoon.waterreaction.sync;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class UploadCleanupPolicyTest {
    @Test
    public void successCompletesOnlyAfterArchiveDeletion() {
        assertFalse(UploadCleanupPolicy.mayDeleteQueueRow(
                UploadDecision.SUCCESS, false));
        assertTrue(UploadCleanupPolicy.mayDeleteQueueRow(
                UploadDecision.SUCCESS, true));
    }

    @Test
    public void retryAndBlockedNeverDeleteArchiveOrQueueRow() {
        for (UploadDecision decision : new UploadDecision[] {
                UploadDecision.RETRY, UploadDecision.BLOCKED}) {
            assertFalse(UploadCleanupPolicy.mayDeleteArchive(decision));
            assertFalse(UploadCleanupPolicy.mayDeleteQueueRow(decision, true));
        }
    }
}
