package org.hiddenmoon.waterreaction;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class BuildContractTest {
    @Test
    public void packageNameIsPermanent() {
        assertEquals("org.hiddenmoon.waterreaction", BuildConfig.APPLICATION_ID);
    }
}
