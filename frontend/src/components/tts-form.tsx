"use client";

import { useState, useEffect, ChangeEvent } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { api } from "@/lib/api";
import { audioBufferToWav } from "@/lib/audio-utils";
import { toast } from "sonner";

interface VoiceOption {
  id: string;
  name: string;
  // add any other properties that your voice objects have
}

export function TTSForm() {
  const [text, setText] = useState("");
  const [voice, setVoice] = useState("af_heart");
  const [speed, setSpeed] = useState(1.0);
  const [useGpu, setUseGpu] = useState(true);
  const [voices, setVoices] = useState<VoiceOption[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  useEffect(() => {
    // Fetch available voices
    api
      .getVoices()
      .then((voices: VoiceOption[]) => {
        setVoices(voices);
      })
      .catch((error: Error) => {
        console.error("Failed to load voices:", error);
        toast.error("Failed to load voices");
      });
  }, []);

  const handleSubmit = async () => {
    try {
      setIsLoading(true);
      const response = await api.generateSpeech({
        text,
        voice,
        speed,
        use_gpu: useGpu,
      });

      // Create an audio context
      const AudioContextClass =
        window.AudioContext || (window as any).webkitAudioContext;
      const audioContext = new AudioContextClass();

      // Create a Float32Array from the response data
      const audioData = new Float32Array(response.audio);

      // Create an AudioBuffer
      const audioBuffer = audioContext.createBuffer(
        1,
        audioData.length,
        response.sample_rate
      );
      audioBuffer.copyToChannel(audioData, 0);

      // Convert to 16-bit PCM
      const pcmData = new Int16Array(audioData.length);
      const scale = 0x7fff; // Max value for 16-bit signed integer

      for (let i = 0; i < audioData.length; i++) {
        const s = Math.max(-1, Math.min(1, audioData[i]));
        pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }

      // Create WAV header
      const wavHeader = new ArrayBuffer(44);
      const view = new DataView(wavHeader);

      // "RIFF" chunk descriptor
      view.setUint32(0, 0x46464952, true); // "RIFF" in ASCII
      view.setUint32(4, 36 + pcmData.length * 2, true); // File length
      view.setUint32(8, 0x45564157, true); // "WAVE" in ASCII

      // "fmt " sub-chunk
      view.setUint32(12, 0x20746d66, true); // "fmt " in ASCII
      view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
      view.setUint16(20, 1, true); // AudioFormat (1 for PCM)
      view.setUint16(22, 1, true); // NumChannels (1 for mono)
      view.setUint32(24, response.sample_rate, true); // SampleRate
      view.setUint32(28, response.sample_rate * 2, true); // ByteRate
      view.setUint16(32, 2, true); // BlockAlign
      view.setUint16(34, 16, true); // BitsPerSample

      // "data" sub-chunk
      view.setUint32(36, 0x61746164, true); // "data" in ASCII
      view.setUint32(40, pcmData.length * 2, true); // Subchunk2Size

      // Combine header and PCM data
      const wavBlob = new Blob([wavHeader, pcmData.buffer], {
        type: "audio/wav",
      });

      const url = URL.createObjectURL(wavBlob);
      setAudioUrl(url);
      toast.success("Speech generated successfully!");
    } catch (error) {
      console.error("Failed to generate speech:", error);
      if (error instanceof Error) {
        toast.error(`Failed to generate speech: ${error.message}`);
      } else {
        toast.error("Failed to generate speech: An unknown error occurred");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleTextChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    // Set the text to the value of the textarea
    setText(e.target.value);
  };

  return (
    <div className="space-y-6 max-w-2xl mx-auto p-6">
      <div className="space-y-2">
        <label className="text-sm font-medium">Text</label>
        <Textarea
          value={text}
          onChange={handleTextChange}
          placeholder="Enter text to convert to speech..."
          className="min-h-[100px]"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Voice</label>
          <Select value={voice} onValueChange={setVoice}>
            <SelectTrigger>
              <SelectValue placeholder="Select a voice" />
            </SelectTrigger>
            <SelectContent>
              {voices.map((v) => (
                <SelectItem key={v.id} value={v.id}>
                  {v.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Speed ({speed}x)</label>
          <Slider
            value={[speed]}
            onValueChange={(values: number[]) => setSpeed(values[0])}
            min={0.5}
            max={2}
            step={0.1}
          />
        </div>
      </div>

      <div className="flex items-center space-x-2">
        <Switch checked={useGpu} onCheckedChange={setUseGpu} id="gpu-mode" />
        <label htmlFor="gpu-mode" className="text-sm font-medium">
          Use GPU Acceleration
        </label>
      </div>

      <Button
        onClick={handleSubmit}
        disabled={!text || isLoading}
        className="w-full"
      >
        {isLoading ? "Generating..." : "Generate Speech"}
      </Button>

      {audioUrl && (
        <div className="mt-4">
          <audio
            controls
            src={audioUrl}
            className="w-full"
            onEnded={() => {
              URL.revokeObjectURL(audioUrl);
              setAudioUrl(null);
            }}
          />
        </div>
      )}
    </div>
  );
}
