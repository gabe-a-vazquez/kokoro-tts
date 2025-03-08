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
import { api, VoiceOption } from "@/lib/api";
import { toast } from "sonner";

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

      .then(setVoices)
      .catch((error: Error) => {
        console.error("Failed to load voices:", error);
        toast.error("Failed to load voices");
      });
  }, []);

  const handleSubmit = async () => {
    try {
      setIsLoading(true);
      const audioBuffer = await api.generateSpeech({
        text,
        voice,
        speed,
        use_gpu: useGpu,
      });

      // Convert ArrayBuffer to Blob and create URL
      const blob = new Blob([audioBuffer], { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
      toast.success("Speech generated successfully!");
    } catch (error: unknown) {
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
                  {v.emoji} {v.name}
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
