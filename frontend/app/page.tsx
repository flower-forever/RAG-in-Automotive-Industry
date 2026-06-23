'use client';
import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Message {
  role: 'user' | 'ai';
  text: string;
  thinking?: string;
  confidence?: number;
  cited?: any[];
  error?: boolean;
}

export default function Home() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleAsk = async () => {
    if (!query.trim()) return;
    const currentQuery = query;
    setQuery('');
    setMessages(prev => [...prev, { role: 'user', text: currentQuery }]);
    setLoading(true);
    
    // Add an empty AI message to stream into
    setMessages(prev => [...prev, { role: 'ai', text: '', thinking: '' }]);

    try {
      const res = await fetch('http://localhost:8000/api/ask_stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: currentQuery })
      });

      if (!res.body) throw new Error("No body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let fullText = "";
      let metadataStr = "";
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        fullText += chunk;

        // Handle Critique Fail
        if (fullText.includes('__CRITIQUE_FAIL__')) {
          setMessages(prev => {
            const newMessages = [...prev];
            newMessages[newMessages.length - 1] = {
              ...newMessages[newMessages.length - 1],
              text: "I don't have enough information in my knowledge base to answer this.",
              thinking: '',
              error: true
            };
            return newMessages;
          });
          break; // Stop processing
        }

        // Handle Metadata
        if (fullText.includes('__METADATA__:')) {
          const parts = fullText.split('__METADATA__:');
          fullText = parts[0];
          try {
            const metadata = JSON.parse(parts[1]);
            setMessages(prev => {
              const newMessages = [...prev];
              newMessages[newMessages.length - 1] = {
                ...newMessages[newMessages.length - 1],
                confidence: metadata.confidence,
                cited: metadata.cited
              };
              return newMessages;
            });
          } catch (e) {}
          break; // Stream ends here
        }

        // Parse expanded queries (ignore visually for now, or could log them)
        if (fullText.includes('__EXPANDED_QUERIES__:')) {
          const parts = fullText.split('\n\n', 2);
          if (parts.length > 1) {
            fullText = parts[1]; // Remove it from text
          }
        }

        // Parse Thinking vs Text
        let parsedText = fullText;
        let parsedThinking = '';
        
        const thinkingMatch = fullText.match(/<thinking>(.*?)<\/thinking>/s);
        if (thinkingMatch) {
          parsedThinking = thinkingMatch[1].trim();
          parsedText = fullText.replace(/<thinking>.*?<\/thinking>/s, '').trim();
        } else if (fullText.includes('<thinking>')) {
          parsedThinking = fullText.replace('<thinking>', '').trim();
          parsedText = '';
        }

        parsedText = parsedText.replace(/<answer>/g, '').replace(/<\/answer>/g, '').trim();

        // Update state
        setMessages(prev => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1] = {
            ...newMessages[newMessages.length - 1],
            text: parsedText,
            thinking: parsedThinking
          };
          return newMessages;
        });
      }
    } catch (e) {
      setMessages(prev => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = {
          ...newMessages[newMessages.length - 1],
          text: 'Error connecting to server.',
          error: true
        };
        return newMessages;
      });
    }
    
    setLoading(false);
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const files = Array.from(e.target.files);
    
    if (files.length > 3) {
      alert('Max 3 files allowed.');
      return;
    }
    for (const f of files) {
      if (f.size > 5 * 1024 * 1024) {
        alert(`File ${f.name} exceeds 5MB limit.`);
        return;
      }
    }

    const formData = new FormData();
    files.forEach(f => formData.append('files', f));

    alert('Uploading and rebuilding index. This may take a moment...');
    try {
      const res = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData
      });
      if (res.ok) alert('Success! Index rebuilt.');
      else alert('Failed to upload.');
    } catch (e) {
      alert('Error connecting to server.');
    }
  };

  return (
    <main className="min-h-screen bg-slate-50 dark:bg-darkspace text-slate-900 dark:text-white p-8 font-sans flex flex-col transition-colors duration-300">
      <header className="flex justify-between items-center mb-8 pb-4 border-b border-slate-200 dark:border-slate-800 backdrop-blur-md sticky top-0 z-10">
        <h1 className="text-3xl font-mono font-bold text-cybercyan flex items-center gap-2">
          <span className="w-4 h-4 bg-cybercyan inline-block rounded-sm animate-pulse"></span>
          SecureOps RAG
        </h1>
        <div>
          <label className="bg-cybercyan text-white px-5 py-2.5 rounded-md cursor-pointer hover:bg-cyan-600 hover:shadow-[0_0_15px_rgba(6,182,212,0.5)] transition-all font-mono text-sm shadow-md">
            + Upload Docs
            <input type="file" multiple accept=".pdf,.txt" className="hidden" onChange={handleUpload} />
          </label>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto mb-6 space-y-6 max-w-4xl mx-auto w-full px-4 scroll-smooth">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-slate-400 dark:text-slate-500 font-mono space-y-4">
            <div className="w-16 h-16 border-4 border-slate-200 dark:border-slate-800 border-t-cybercyan rounded-full animate-spin"></div>
            <p>System Online. Awaiting queries...</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`p-5 rounded-2xl max-w-[85%] shadow-sm ${m.role === 'user' ? 'bg-slate-200 dark:bg-slate-800 text-right rounded-br-none' : 'bg-white dark:bg-slate-900 border border-slate-100 dark:border-slate-800 rounded-bl-none'}`}>
              
              {/* User Message */}
              {m.role === 'user' && <p className="whitespace-pre-wrap leading-relaxed">{m.text}</p>}
              
              {/* AI Message */}
              {m.role === 'ai' && (
                <div className="flex flex-col gap-3">
                  {/* Thinking Block */}
                  {m.thinking && (
                    <details className="bg-slate-50 dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700/50 text-sm overflow-hidden group">
                      <summary className="p-3 cursor-pointer font-mono text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors list-none flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-slate-400 group-open:bg-cybercyan"></span>
                        Thinking Process
                      </summary>
                      <div className="p-3 pt-0 border-t border-slate-200 dark:border-slate-700/50 mt-1 text-slate-600 dark:text-slate-400 whitespace-pre-wrap font-mono">
                        {m.thinking}
                      </div>
                    </details>
                  )}
                  
                  {/* Markdown Rendered Text */}
                  {m.text && (
                    <div className="prose dark:prose-invert max-w-none prose-p:leading-relaxed prose-pre:bg-slate-100 dark:prose-pre:bg-slate-800 prose-pre:border prose-pre:border-slate-200 dark:prose-pre:border-slate-700">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {m.text}
                      </ReactMarkdown>
                    </div>
                  )}

                  {/* Confidence & Citations */}
                  {m.confidence !== undefined && !m.error && (
                    <div className="mt-3 pt-3 border-t border-slate-100 dark:border-slate-800 text-left flex flex-wrap gap-2 items-center">
                      <span className={`text-xs font-mono inline-flex items-center gap-1 px-2.5 py-1 rounded-md ${m.confidence > 0.7 ? 'bg-securegreen/10 text-securegreen border border-securegreen/20' : 'bg-alertrose/10 text-alertrose border border-alertrose/20'}`}>
                        {m.confidence > 0.7 ? '✓' : '⚠'} Confidence: {(m.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && messages.length > 0 && messages[messages.length-1].role === 'user' && (
          <div className="flex justify-start">
            <div className="p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-100 dark:border-slate-800 rounded-bl-none shadow-sm flex items-center gap-3">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-cybercyan rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                <div className="w-2 h-2 bg-cybercyan rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                <div className="w-2 h-2 bg-cybercyan rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
              </div>
              <span className="text-sm font-mono text-slate-500 dark:text-slate-400">Initializing connection...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="max-w-4xl mx-auto w-full relative group">
        <div className="absolute -inset-1 bg-gradient-to-r from-cybercyan to-securegreen rounded-xl blur opacity-25 group-hover:opacity-40 transition duration-1000 group-hover:duration-200"></div>
        <div className="relative flex gap-3 bg-white dark:bg-slate-900 p-2 rounded-xl border border-slate-200 dark:border-slate-800 shadow-xl">
          <input 
            type="text" 
            value={query} 
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleAsk()}
            placeholder="Query SecureOps database..." 
            className="flex-1 p-3 bg-transparent focus:outline-none font-sans text-slate-900 dark:text-white placeholder-slate-400"
          />
          <button 
            onClick={handleAsk} 
            disabled={loading} 
            className="bg-slate-900 dark:bg-white text-white dark:text-slate-900 px-8 py-3 rounded-lg font-bold hover:bg-slate-800 dark:hover:bg-slate-200 transition-colors disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </div>
    </main>
  );
}
