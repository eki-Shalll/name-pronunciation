import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HOST = '127.0.0.1';
const PORT = Number(process.env.PORT || 8787);
const API_KEY = process.env.OPENAI_API_KEY;
const __dirname = dirname(fileURLToPath(import.meta.url));
const allowedVoices = new Set(['alloy','ash','ballad','coral','echo','fable','nova','onyx','sage','shimmer','verse','marin','cedar']);

function json(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(body));
}

async function readJson(req) {
  let raw = '';
  for await (const chunk of req) {
    raw += chunk;
    if (raw.length > 50_000) throw new Error('Request too large');
  }
  return JSON.parse(raw || '{}');
}

const server = http.createServer(async (req, res) => {
  try {
    if (req.method === 'GET' && (req.url === '/' || req.url === '/openai-lab.html')) {
      const html = await readFile(join(__dirname, 'openai-lab.html'));
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' });
      res.end(html);
      return;
    }

    if (req.method === 'POST' && req.url === '/api/tts') {
      if (!API_KEY) {
        json(res, 500, { error: 'OPENAI_API_KEY is not set in the local server environment.' });
        return;
      }

      const body = await readJson(req);
      const text = String(body.text || '').trim();
      const voice = String(body.voice || 'marin');
      const instructions = String(body.instructions || '').trim();
      const speed = Number(body.speed ?? 1);

      if (!text || text.length > 4096) {
        json(res, 400, { error: 'Text must be between 1 and 4096 characters.' });
        return;
      }
      if (!allowedVoices.has(voice)) {
        json(res, 400, { error: 'Unsupported voice.' });
        return;
      }
      if (!Number.isFinite(speed) || speed < 0.25 || speed > 4) {
        json(res, 400, { error: 'Speed must be between 0.25 and 4.0.' });
        return;
      }

      const payload = {
        model: 'gpt-4o-mini-tts',
        input: text,
        voice,
        speed,
        response_format: 'wav'
      };
      if (instructions) payload.instructions = instructions;

      const response = await fetch('https://api.openai.com/v1/audio/speech', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${API_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        let message = `OpenAI API error ${response.status}`;
        try {
          const data = await response.json();
          message = data?.error?.message || message;
        } catch {}
        json(res, response.status, { error: message });
        return;
      }

      const audio = Buffer.from(await response.arrayBuffer());
      res.writeHead(200, {
        'Content-Type': 'audio/wav',
        'Content-Length': audio.length,
        'Cache-Control': 'no-store'
      });
      res.end(audio);
      return;
    }

    json(res, 404, { error: 'Not found' });
  } catch (error) {
    json(res, 500, { error: error instanceof Error ? error.message : String(error) });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`OpenAI TTS Lab: http://${HOST}:${PORT}`);
  if (!API_KEY) console.warn('OPENAI_API_KEY is not set. Set it before generating audio.');
});
