import { TTSForm } from "@/components/tts-form";

export default function Home() {
  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto py-10">
        <h1 className="text-4xl font-bold text-center mb-8">Kokoro TTS</h1>
        <p className="text-center text-muted-foreground mb-10">
          An open-weight TTS model with 82 million parameters
        </p>
        <TTSForm />
      </div>
    </main>
  );
}
