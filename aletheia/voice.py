from __future__ import annotations

import platform
import shutil
import subprocess


class VoiceOutput:
    """Host-level speech output; it does not load a pretrained Aletheia model."""

    def speak(self, text: str) -> dict:
        text = " ".join(str(text).split()).strip()
        if not text:
            return {"spoken": False, "reason": "empty_text"}

        system = platform.system()
        if system == "Windows":
            # Do not interpolate generated text into PowerShell source. Passing it
            # through stdin keeps quotes, backticks, and PowerShell metacharacters
            # data rather than code.
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$text=[Console]::In.ReadToEnd(); $s.Speak($text)"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                input=text,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            return {"spoken": result.returncode == 0, "backend": "windows_sapi", "error": result.stderr.strip()}

        if system == "Darwin" and shutil.which("say"):
            result = subprocess.run(["say"], input=text, capture_output=True, text=True, encoding="utf-8")
            return {"spoken": result.returncode == 0, "backend": "macos_say", "error": result.stderr.strip()}

        if shutil.which("espeak"):
            result = subprocess.run(["espeak"], input=text, capture_output=True, text=True, encoding="utf-8")
            return {"spoken": result.returncode == 0, "backend": "espeak", "error": result.stderr.strip()}

        return {"spoken": False, "reason": "no_system_tts_backend"}


    def play_waveform(self, waveform, *, sample_rate: int = 16000) -> dict:
        """Play a mono float waveform through the optional sounddevice backend."""
        try:
            import sounddevice as sd
        except Exception as exc:
            return {"played": False, "reason": "sounddevice_unavailable", "error": repr(exc)}
        try:
            sd.play(waveform.detach().float().cpu().numpy(), samplerate=int(sample_rate), blocking=True)
            return {"played": True, "backend": "sounddevice"}
        except Exception as exc:
            return {"played": False, "backend": "sounddevice", "error": repr(exc)}
