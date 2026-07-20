package org.hiddenmoon.waterreaction;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class UiTokenTest {
    @Test
    public void archivePaletteUsesHighContrastTokens() {
        assertEquals(0xFF111111, UiPalette.INK);
        assertEquals(0xFFFFFFFF, UiPalette.PAPER);
        assertEquals(0xFFD9FF3F, UiPalette.SIGNAL_YELLOW);
        assertEquals(0xFF00A7B5, UiPalette.SIGNAL_CYAN);
    }

    @Test
    public void reducedMotionStillAllowsStaticSurface() {
        assertTrue(UiPalette.staticMotionDurationMs() == 0L);
    }
}
