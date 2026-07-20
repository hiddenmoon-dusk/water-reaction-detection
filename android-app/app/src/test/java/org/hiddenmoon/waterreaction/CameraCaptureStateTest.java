package org.hiddenmoon.waterreaction;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class CameraCaptureStateTest {
    @Test
    public void pendingUriSurvivesActivityRecreation() {
        CameraCaptureState before = new CameraCaptureState();
        before.begin("content://media/photo/42");

        CameraCaptureState after = new CameraCaptureState();
        after.restore(before.pendingUri());

        assertEquals("content://media/photo/42", after.consume(true, false));
        assertNull(after.pendingUri());
    }

    @Test
    public void acceptsPhotoWrittenByCameraEvenWhenResultCodeIsNotOk() {
        CameraCaptureState state = new CameraCaptureState();
        state.begin("content://media/photo/43");

        assertEquals("content://media/photo/43", state.consume(false, true));
    }

    @Test
    public void rejectsCancelledEmptyCapture() {
        CameraCaptureState state = new CameraCaptureState();
        state.begin("content://media/photo/44");

        assertNull(state.consume(false, false));
        assertNull(state.pendingUri());
    }

    @Test
    public void archiveSuccessDeletesOnlyCameraSource() {
        assertTrue(SourceCleanupPolicy.shouldDeleteCapture("camera", true));
        assertFalse(SourceCleanupPolicy.shouldDeleteCapture("gallery", true));
    }

    @Test
    public void archiveFailureKeepsCameraForAnotherSaveAttempt() {
        assertFalse(SourceCleanupPolicy.shouldDeleteCapture("camera", false));
    }
}
