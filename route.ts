// File: app/api/chat/route.ts
// HidupAI Memory Engine v2 — PROD READY (timeout-friendly)

import { createClient } from '@supabase/supabase-js';
import { NextResponse } from 'next/server';
import { jwtDecode } from 'jwt-decode';
import { GoogleGenerativeAI } from '@google/generative-ai';

export const dynamic = 'force-dynamic';

/* =========================================================
 * SUPABASE CLIENT
 * =======================================================*/

const supabase = createClient(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface JwtPayload {
  email: string;
}

interface HidupAIUserRow {
  id: string;
  is_premium: boolean;
  usage_today: number | null;
  last_used: string | null;
  long_term_memory: string | null;
  preferred_mode: string | null;
  weekly_goal: string | null;
  last_question: string | null;
  last_response: string | null;
}

interface MemoryEmbeddingRow {
  content: string;
  embedding: number[];
}

type MemorySourceType = 'chat' | 'episodic';

interface MemoryEmbeddingInsert {
  user_id: string;
  source_type: MemorySourceType;
  source_id: string | null;
  content: string;
  embedding: number[];
}

/* =========================================================
 * AI CONFIG — OPENAI + GEMINI
 * =======================================================*/

const OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions';
const OPENAI_EMBED_URL = 'https://api.openai.com/v1/embeddings';

const OPENAI_MAIN_MODEL =
  process.env.OPENAI_MAIN_MODEL || 'gpt-4o-mini';

const OPENAI_SUMMARIZER_MODEL =
  process.env.OPENAI_SUMMARIZER_MODEL || 'gpt-4o-mini';

const OPENAI_EMBED_MODEL =
  process.env.OPENAI_EMBED_MODEL || 'text-embedding-3-small';

const GEMINI_MODEL =
  process.env.GEMINI_MODEL || 'gemini-1.5-flash-latest';

// Mode provider: openai | gemini | hybrid
const AI_PROVIDER =
  (process.env.AI_PROVIDER as 'openai' | 'gemini' | 'hybrid') || 'hybrid';

const genAI = process.env.GOOGLE_API_KEY
  ? new GoogleGenerativeAI(process.env.GOOGLE_API_KEY)
  : null;

/* =========================================================
 *  HUMAN MIND STACK PROMPT ENGINE
 * =======================================================*/

const buildBasePrompt = (
  name: string,
  goal: string,
  memoryBlock: string
): string => {
  return `
Kamu adalah HidupAI™, sahabat hidup ${name}.

TENTANG HIDUPAI:
- Kamu human-centered: fokusmu memahami manusia, bukan sekadar memberi jawaban.
- Kamu empatik, reflektif, tenang, dan tidak menghakimi.
- Kamu tidak menyebut diri sebagai AI, chatbot, atau model. Cukup "HidupAI".

KONTEKS HIDUP ${name}:
${memoryBlock || '- Belum ada catatan hidup yang terekam.'}

PRINSIP CARA KERJA (Human Mind Stack):

1) EMOSI  
   - Baca emosi di balik kata-kata ${name}.
   - Validasi perasaan sebelum memberi solusi.

2) PIKIRAN & BIAS  
   - Bantu ${name} melihat pola pikir yang tidak sehat.
   - Luruskan generalisasi berlebihan dengan lembut.

3) MOTIVASI  
   - Dukung autonomy, competence, relatedness.

4) MAKNA  
   - Hubungkan keputusan dengan nilai hidup ${name}.

5) RUANG AMAN  
   - Bahasa lembut, tidak menghakimi.
   - Bila topik berat (trauma, self-harm), anjurkan bicara dengan manusia terpercaya.

CATATAN PENTING TENTANG MEMORI:
- Anggap semua catatan di atas sebagai hal-hal yang kamu ingat tentang ${name}.
- Jika ${name} bertanya soal "obrolan kemarin" atau percakapan sebelumnya,
  gunakan konteks memori dan percakapan terakhir untuk menjawab seolah kamu mengingatnya.
- Jangan menjawab "aku tidak bisa mengingat percakapan sebelumnya"
  selama masih ada catatan atau memori yang bisa dipakai.

GAYA BAHASA:
- Tenang, dewasa, hangat, tidak robotik.
- Boleh pakai emoji ✨🌱🤍 secukupnya.
- Gunakan kalimat pendek–sedang, seolah kamu ngobrol 1:1.

MISI:
- Bantu ${name} memahami dirinya, bukan sekadar menjawab pertanyaan.

FORMAT JAWABAN:
- Jawab dalam bahasa Indonesia.
- Gunakan paragraf-paragraf pendek.
- JANGAN gunakan Markdown (tanpa **bold**, tanda *italics*, heading #, atau bullet list dengan "-").
- Jangan gunakan blok kode.
- Jika perlu memberi langkah-langkah, gunakan format:
  1) ...
  2) ...
  3) ...
`.trim();
};

const buildModeAddon = (name: string, mode: string): string => {
  switch (mode) {
    case 'pagi':
      return `
PAGI HARI ☀️
- Sambut ${name} dengan ringan dan positif.
- Ajak set niat harian sederhana.
- Hindari to-do berlebihan.
`.trim();
    case 'mentok':
      return `
MENTOK / BUNTU 🌱
- Tugas utama: turunkan beban pikiran.
- Ajukan pertanyaan reflektif yang lembut.
- Beri langkah kecil yang realistis.
`.trim();
    case 'sedih':
      return `
SEDIH / LETIH 🤍
- Peluk lewat kata.
- Validasi emosi.
- Hindari toxic positivity.
`.trim();
    case 'sukses':
      return `
SUKSES 🎉
- Rayakan pencapaian ${name}.
- Ajak refleksi proses dan usaha.
- Perkuat self-worth.
`.trim();
    default:
      return `
MODE UMUM
- Respon berdasarkan emosi dari pesan user.
`.trim();
  }
};

const getPrompt = (
  name: string,
  mode: string,
  goal: string,
  memoryBlock: string
): ChatMessage => ({
  role: 'system',
  content: `${buildBasePrompt(
    name,
    goal,
    memoryBlock
  )}\n\n${buildModeAddon(name, mode)}`,
});

/* =========================================================
 *  AI CALLERS — OPENAI / GEMINI / HYBRID
 * =======================================================*/

async function callOpenAI(
  systemPrompt: ChatMessage,
  messages: ChatMessage[]
): Promise<string> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY missing');
  }

  const res = await fetch(OPENAI_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: OPENAI_MAIN_MODEL,
      temperature: 0.8,
      messages: [systemPrompt, ...messages],
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    console.error('[OpenAI] error', res.status, text);
    throw new Error(`OpenAI error ${res.status}`);
  }

  const json = await res.json();
  const msg: string | undefined =
    json.choices?.[0]?.message?.content?.trim();
  if (!msg) throw new Error('OpenAI returned empty');

  return msg;
}

async function callGemini(
  systemPrompt: ChatMessage,
  messages: ChatMessage[]
): Promise<string> {
  if (!genAI) throw new Error('GOOGLE_API_KEY missing');

  const model = genAI.getGenerativeModel({ model: GEMINI_MODEL });

  const combined = [systemPrompt, ...messages]
    .map((m) => `${m.role.toUpperCase()}:\n${m.content}`)
    .join('\n\n');

  const res = await model.generateContent({
    contents: [
      {
        role: 'user',
        parts: [{ text: combined }],
      },
    ],
  });

  const text = res.response.text().trim();
  if (!text) throw new Error('Gemini returned empty');

  return text;
}

async function callMainModel(
  systemPrompt: ChatMessage,
  messages: ChatMessage[]
): Promise<string> {
  if (AI_PROVIDER === 'gemini') return callGemini(systemPrompt, messages);
  if (AI_PROVIDER === 'openai') return callOpenAI(systemPrompt, messages);

  // HYBRID MODE
  try {
    return await callOpenAI(systemPrompt, messages);
  } catch (err) {
    console.warn('[HidupAI] OpenAI gagal, fallback ke Gemini', err);
    return callGemini(systemPrompt, messages);
  }
}

/* =========================================================
 *  FORMAT NORMALIZER
 * =======================================================*/

function stripMarkdown(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, '')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/^\s*-\s+/gm, '• ')
    .replace(/[ \t]+\n/g, '\n')
    .trim();
}

/* =========================================================
 *  EMBEDDING ENGINE (JSONB)
 * =======================================================*/

async function getEmbedding(text: string): Promise<number[] | null> {
  if (!process.env.OPENAI_API_KEY) return null;
  const cleaned = text.trim();
  if (!cleaned) return null;

  try {
    const res = await fetch(OPENAI_EMBED_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: OPENAI_EMBED_MODEL,
        input: cleaned,
      }),
    });

    if (!res.ok) {
      console.error('[embedding] OpenAI error', await res.text());
      return null;
    }

    const json = await res.json();
    const emb: number[] | undefined =
      json.data?.[0]?.embedding || undefined;

    return Array.isArray(emb) ? emb : null;
  } catch (err) {
    console.error('[embedding] fatal error:', err);
    return null;
  }
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (!a.length || !b.length || a.length !== b.length) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (!na || !nb) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

async function retrieveSimilarMemories(
  userId: string | null,
  query: string
): Promise<string> {
  if (!userId || !query.trim()) return '';
  const queryEmbedding = await getEmbedding(query);
  if (!queryEmbedding) return '';

  try {
    const { data: rows, error } = await supabase
      .from('memory_embeddings')
      .select('content, embedding')
      .eq('user_id', userId)
      .order('created_at', { ascending: false })
      .limit(200);

    if (error || !rows || rows.length === 0) return '';

    const typedRows = rows as unknown as MemoryEmbeddingRow[];

    const scored = typedRows
      .map((row) => {
        const emb = row.embedding;
        if (!Array.isArray(emb)) return null;
        const score = cosineSimilarity(queryEmbedding, emb);
        return { content: row.content, score };
      })
      .filter(
        (item): item is { content: string; score: number } =>
          item !== null && item.score > 0.65
      );

    if (!scored.length) return '';

    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, 5);

    return top.map((m) => `- ${m.content}`).join('\n');
  } catch (err) {
    console.error('[retrieval] error:', err);
    return '';
  }
}

/* =========================================================
 *  MEMORY ENGINE — INSIGHT & EPISODIC
 * =======================================================*/

async function extractInsight(
  name: string,
  prev: string,
  userMsg: string,
  aiMsg: string
): Promise<string | null> {
  if (!process.env.OPENAI_API_KEY) return null;
  if (!userMsg || aiMsg.length < 40) return null;

  try {
    const res = await fetch(OPENAI_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: OPENAI_SUMMARIZER_MODEL,
        temperature: 0.2,
        messages: [
          {
            role: 'system',
            content: `
Tuliskan SATU bullet insight tentang ${name}.
Fokus: nilai hidup, kekhawatiran, harapan, atau pola pikir.
Singkat, netral, tanpa emoji, tanpa sapaan.`.trim(),
          },
          {
            role: 'user',
            content: `
Memori sebelumnya:
${prev}

Pesan user:
${userMsg}

Jawaban HidupAI:
${aiMsg}
`.trim(),
          },
        ],
      }),
    });

    if (!res.ok) return null;

    const json = await res.json();
    const insight =
      json.choices?.[0]?.message?.content
        ?.trim()
        ?.replace(/^[-•]\s*/, '') || '';

    return insight ? `- ${insight}` : null;
  } catch {
    return null;
  }
}

type EpisodicEvent = {
  summary: string;
  tags: string[];
};

async function extractEpisodicEvent(
  name: string,
  userMsg: string,
  aiMsg: string
): Promise<EpisodicEvent | null> {
  if (!process.env.OPENAI_API_KEY) return null;
  if (!userMsg || aiMsg.length < 40) return null;

  try {
    const res = await fetch(OPENAI_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: OPENAI_SUMMARIZER_MODEL,
        temperature: 0.2,
        messages: [
          {
            role: 'system',
            content: `
Ringkas interaksi berikut sebagai SATU kejadian hidup ${name}.
Berikan:
1) Ringkasan singkat (maks 2 kalimat).
2) 3-5 tag kata kunci (tanpa emoji).

Format:
RINGKASAN: ...
TAGS: tag1, tag2, tag3
`.trim(),
          },
          {
            role: 'user',
            content: `
Pesan ${name}:
${userMsg}

Jawaban HidupAI:
${aiMsg}
`.trim(),
          },
        ],
      }),
    });

    if (!res.ok) return null;
    const json = await res.json();
    const raw: string =
      json.choices?.[0]?.message?.content?.trim() || '';

    if (!raw) return null;

    const summaryMatch = raw.match(/RINGKASAN:\s*(.+)/i);
    const tagsMatch = raw.match(/TAGS:\s*(.+)/i);

    const summary = summaryMatch?.[1]?.trim() || '';
    const tagsRaw = tagsMatch?.[1] || '';
    const tags = tagsRaw
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean);

    if (!summary) return null;

    return { summary, tags };
  } catch (err) {
    console.error('[episodic] error:', err);
    return null;
  }
}

/**
 * Memory update dijalankan di background (tidak di-await di flow utama)
 */
async function updateMemoriesInBackground(params: {
  name: string;
  email: string;
  userId: string | null;
  memorySummary: string;
  lastUserMessage: string;
  aiMessage: string;
}) {
  const {
    name,
    email,
    userId,
    memorySummary,
    lastUserMessage,
    aiMessage,
  } = params;

  try {
    // 1) Insight → summary + tabel long_term_memories
    const insight = await extractInsight(
      name,
      memorySummary,
      lastUserMessage,
      aiMessage
    );

    if (insight) {
      const updatedSummary = memorySummary
        ? `${memorySummary}\n${insight}`
        : insight;

      await supabase
        .from('users')
        .update({ long_term_memory: updatedSummary })
        .eq('email', email);

      if (userId) {
        await supabase.from('long_term_memories').insert({
          user_id: userId,
          content: insight,
        });
      }
    }

    // 2) Episodic event + embedding
    if (userId) {
      const episodic = await extractEpisodicEvent(
        name,
        lastUserMessage,
        aiMessage
      );

      if (episodic) {
        const { summary, tags } = episodic;

        const { data: epiRows, error: epiError } = await supabase
          .from('episodic_memories')
          .insert({
            user_id: userId,
            summary,
            raw_text: `${name}: ${lastUserMessage}\nHidupAI: ${aiMessage}`,
            tags,
          })
          .select('id')
          .single();

        const episodicRow = epiRows as { id: string } | null;
        const episodicId =
          !epiError && episodicRow ? episodicRow.id : null;

        const [userEmb, epiEmb] = await Promise.all([
          getEmbedding(lastUserMessage),
          getEmbedding(summary),
        ]);

        const inserts: MemoryEmbeddingInsert[] = [];

        if (userEmb) {
          inserts.push({
            user_id: userId,
            source_type: 'chat',
            source_id: null,
            content: lastUserMessage,
            embedding: userEmb,
          });
        }

        if (episodicId && epiEmb) {
          inserts.push({
            user_id: userId,
            source_type: 'episodic',
            source_id: episodicId,
            content: summary,
            embedding: epiEmb,
          });
        }

        if (inserts.length > 0) {
          await supabase.from('memory_embeddings').insert(inserts);
        }
      }
    }
  } catch (err) {
    console.error('[memory update] error:', err);
  }
}

/* =========================================================
 *  ROUTE HANDLER
 * =======================================================*/

type ChatRequestBody = {
  messages: ChatMessage[];
  name: string;
  email: string;
  mode?: string;
};

export async function POST(req: Request) {
  try {
    // Parsing body aman
    const body = (await req.json()) as ChatRequestBody;
    const { messages, name, email, mode: rawMode } = body;

    if (!messages || !Array.isArray(messages) || !name || !email) {
      return NextResponse.json(
        { message: 'Payload tidak lengkap' },
        { status: 400 }
      );
    }

    // Token check
    const token = req.headers.get('Authorization')?.split(' ')[1];
    if (!token) {
      return NextResponse.json(
        { message: 'Token tidak ditemukan' },
        { status: 401 }
      );
    }

    let decodedEmail = '';
    try {
      const decoded = jwtDecode<JwtPayload>(token);
      decodedEmail = decoded.email;
    } catch {
      return NextResponse.json(
        { message: 'Token tidak valid' },
        { status: 401 }
      );
    }

    if (decodedEmail !== email) {
      return NextResponse.json(
        { message: 'Akses tidak sah 🔒' },
        { status: 401 }
      );
    }

    // Ambil user + id (untuk long_term_memories + percakapan terakhir)
    const { data: user, error } = await supabase
      .from('users')
      .select(
        'id, is_premium, usage_today, last_used, long_term_memory, preferred_mode, weekly_goal, last_question, last_response'
      )
      .eq('email', email)
      .single();

    if (error || !user) {
      return NextResponse.json(
        { message: 'Akun tidak ditemukan' },
        { status: 404 }
      );
    }

    const typedUser = user as HidupAIUserRow;
    const userId: string | null = typedUser.id || null;
    const lastQuestion = typedUser.last_question || '';
    const lastResponse = typedUser.last_response || '';

    /* ===== KUOTA ===== */

    const today = new Date().toISOString().split('T')[0];

    if (typedUser.last_used !== today) {
      await supabase
        .from('users')
        .update({ usage_today: 1, last_used: today })
        .eq('email', email);
    } else if (!typedUser.is_premium && (typedUser.usage_today ?? 0) >= 5) {
      return NextResponse.json(
        {
          message:
            'Batas penggunaan gratis sudah 5x hari ini 😅\nCoba lagi besok ya! 🚀',
        },
        { status: 200 }
      );
    } else if (!typedUser.is_premium) {
      await supabase
        .from('users')
        .update({
          usage_today: (typedUser.usage_today ?? 0) + 1,
        })
        .eq('email', email);
    }

    /* ===== MAIN CHAT ===== */

    const mode = rawMode || typedUser.preferred_mode || '';
    const goal = typedUser.weekly_goal || '';
    const memorySummary: string = typedUser.long_term_memory || '';

    // Ambil beberapa memori panjang terakhir
    let recentMemoryBlock = '';
    try {
      if (userId) {
        const { data: rows, error: memError } = await supabase
          .from('long_term_memories')
          .select('content')
          .eq('user_id', userId)
          .order('created_at', { ascending: false })
          .limit(5);

        if (!memError && rows && rows.length > 0) {
          recentMemoryBlock = rows
            .map((r) => r.content as string)
            .join('\n');
        }
      }
    } catch {
      recentMemoryBlock = '';
    }

    // Percakapan terakhir
    let lastConversationBlock = '';
    if (lastQuestion || lastResponse) {
      const q =
        lastQuestion.length > 500
          ? `${lastQuestion.slice(0, 500)}…`
          : lastQuestion;
      const a =
        lastResponse.length > 500
          ? `${lastResponse.slice(0, 500)}…`
          : lastResponse;

      lastConversationBlock = `
Percakapan terakhir:
- ${name}: ${q || '(tidak ada catatan pertanyaan terakhir)'}
- HidupAI: ${a || '(tidak ada catatan jawaban terakhir)'}
`.trim();
    }

    // Retrieval memori mirip (Memory Engine v2)
    const lastUserMessage =
      messages[messages.length - 1]?.content || '';

    const retrievedBlock = await retrieveSimilarMemories(
      userId,
      lastUserMessage
    );

    const memoryParts: string[] = [];

    if (goal) {
      memoryParts.push(`Tujuan mingguan saat ini: "${goal}"`);
    }

    if (lastConversationBlock) {
      memoryParts.push(lastConversationBlock);
    }

    const lifeMemoryBlock = [memorySummary, recentMemoryBlock]
      .filter(Boolean)
      .join('\n');

    if (lifeMemoryBlock) {
      memoryParts.push(
        `Memori hidup & pola sejauh ini:\n${lifeMemoryBlock}`
      );
    }

    if (retrievedBlock) {
      memoryParts.push(
        `Memori relevan dengan topik ini:\n${retrievedBlock}`
      );
    }

    const memoryBlock = memoryParts.join('\n\n');

    const systemPrompt = getPrompt(name, mode, goal, memoryBlock);

    let aiMessage = '';
    try {
      aiMessage = await callMainModel(systemPrompt, messages);
    } catch (err) {
      console.error('[HidupAI] main model error:', err);
      return NextResponse.json(
        {
          message:
            'Mesin berpikir HidupAI sedang gangguan 😵 Coba sebentar lagi ya.',
        },
        { status: 500 }
      );
    }

    aiMessage = stripMarkdown(aiMessage);

    /* ===== UPDATE LOG (cepat) ===== */

    await supabase
      .from('users')
      .update({
        last_question: lastUserMessage,
        last_response: aiMessage,
        last_interaction: new Date().toISOString(),
      })
      .eq('email', email);

    /* ===== MEMORY ENGINE — DI BACKGROUND (tidak nahan respons) ===== */

    updateMemoriesInBackground({
      name,
      email,
      userId,
      memorySummary,
      lastUserMessage,
      aiMessage,
    });

    return NextResponse.json({ message: aiMessage }, { status: 200 });
  } catch (err) {
    console.error('[HidupAI] fatal error di route /api/chat:', err);
    return NextResponse.json(
      {
        message:
          'HidupAI lagi error internal 😥 Coba beberapa saat lagi ya.',
      },
      { status: 500 }
    );
  }
}
