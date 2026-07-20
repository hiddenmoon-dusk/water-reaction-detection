package org.hiddenmoon.waterreaction;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.Test;

public class ReleaseBuildContractTest {
    private static Path buildFile() {
        Path current = Path.of(System.getProperty("user.dir")).toAbsolutePath();
        for (int depth = 0; current != null && depth < 6; depth++) {
            Path rootFile = current.resolve("app").resolve("build.gradle");
            if (Files.isRegularFile(rootFile)) return rootFile;
            Path moduleFile = current.resolve("build.gradle");
            if (Files.isRegularFile(moduleFile)) {
                String moduleText;
                try {
                    moduleText = new String(
                            Files.readAllBytes(moduleFile), StandardCharsets.UTF_8);
                } catch (Exception ignored) {
                    moduleText = "";
                }
                if (moduleText.contains("com.android.application")) {
                    return moduleFile;
                }
            }
            current = current.getParent();
        }
        throw new AssertionError("无法定位 android-app/app/build.gradle");
    }

    @Test
    public void releaseBuildRequiresExternalSigningConfiguration() throws Exception {
        String text = new String(Files.readAllBytes(buildFile()), StandardCharsets.UTF_8);

        assertTrue(text.contains("release-signing.properties"));
        assertTrue(text.contains("checkReleaseSigning"));
        assertTrue(text.contains("signingConfigs.release"));
        assertFalse(text.contains("signingConfig signingConfigs.debug"));
    }

    @Test
    public void releaseBuildKeepsSupportedAndroidContract() throws Exception {
        String text = new String(Files.readAllBytes(buildFile()), StandardCharsets.UTF_8);

        assertTrue(text.contains("applicationId 'org.hiddenmoon.waterreaction'"));
        assertTrue(text.contains("minSdk 34"));
        assertTrue(text.contains("targetSdk 36"));
    }
}
