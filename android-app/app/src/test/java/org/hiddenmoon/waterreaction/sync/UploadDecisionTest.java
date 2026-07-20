package org.hiddenmoon.waterreaction.sync;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class UploadDecisionTest {
    @Test
    public void mapsServerOutcomes() {
        assertEquals(UploadDecision.SUCCESS, UploadDecision.forHttp(201));
        assertEquals(UploadDecision.SUCCESS, UploadDecision.forHttp(208));
        assertEquals(UploadDecision.RETRY, UploadDecision.forHttp(408));
        assertEquals(UploadDecision.RETRY, UploadDecision.forHttp(429));
        assertEquals(UploadDecision.RETRY, UploadDecision.forHttp(503));
        assertEquals(UploadDecision.BLOCKED, UploadDecision.forHttp(409));
        assertEquals(UploadDecision.BLOCKED, UploadDecision.forHttp(400));
    }
}
