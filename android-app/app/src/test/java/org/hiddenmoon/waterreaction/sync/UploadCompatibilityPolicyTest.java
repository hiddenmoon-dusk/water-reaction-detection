package org.hiddenmoon.waterreaction.sync;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class UploadCompatibilityPolicyTest {
    @Test
    public void onlyLegacyPayloadFailuresAreAutomaticallyRequeued() {
        assertTrue(UploadCompatibilityPolicy.shouldRequeue("invalid_payload"));
        assertFalse(UploadCompatibilityPolicy.shouldRequeue("invalid_archive"));
        assertFalse(UploadCompatibilityPolicy.shouldRequeue("client_release_expired"));
    }
}
