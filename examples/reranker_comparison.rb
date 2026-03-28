require "candle"

device = Candle::Device.best
puts "Device: #{device}"
puts

# ============================================================
# Corpus & Queries
# ============================================================

queries = [
  {
    query: "What is the capital of France?",
    documents: [
      "The capital of France is Paris, known for the Eiffel Tower.",
      "Berlin is the capital of Germany and has a rich history.",
      "London is famous for Big Ben and the Thames river.",
      "France is known for its wine, cheese, and the French Riviera.",
    ]
  },
  {
    query: "How does photosynthesis work?",
    documents: [
      "Photosynthesis converts sunlight, water, and CO2 into glucose and oxygen in plant cells.",
      "The mitochondria is the powerhouse of the cell, producing ATP through cellular respiration.",
      "Plants need sunlight and water to grow, and their leaves are usually green.",
      "Solar panels convert sunlight into electricity using photovoltaic cells.",
    ]
  },
  {
    query: "best practices for password security",
    documents: [
      "Use a password manager and enable two-factor authentication on all accounts.",
      "Strong passwords should be at least 12 characters with mixed case, numbers, and symbols.",
      "The cat sat on the mat and looked out the window at the birds.",
      "Cybersecurity experts recommend rotating passwords every 90 days.",
    ]
  },
]

# ============================================================
# Reranker Models
# ============================================================

rerankers = [
  { name: "MiniLM L-12 (BERT)",       id: "cross-encoder/ms-marco-MiniLM-L-12-v2", arch: "BERT",         params: "33M"  },
  { name: "BGE Base (XLM-RoBERTa)",   id: "BAAI/bge-reranker-base",                arch: "XLM-RoBERTa",  params: "278M" },
  { name: "mxbai Base v1 (DeBERTa)",  id: "mixedbread-ai/mxbai-rerank-base-v1",    arch: "DeBERTa v3",   params: "184M" },
  { name: "GTE (ModernBERT)",         id: "Alibaba-NLP/gte-reranker-modernbert-base", arch: "ModernBERT", params: "149M" },
  { name: "Qwen3 0.6B (Decoder)",     id: "Qwen/Qwen3-Reranker-0.6B",              arch: "Qwen3",        params: "600M" },
]

# ============================================================
# Run comparisons
# ============================================================

# Column widths
name_w = rerankers.map { |r| r[:name].length }.max + 2
doc_w = 62

queries.each_with_index do |q, qi|
  puts "=" * 120
  puts "Query #{qi + 1}: \"#{q[:query]}\""
  puts "=" * 120
  puts

  # Print document legend
  q[:documents].each_with_index do |doc, i|
    puts "  [#{i}] #{doc}"
  end
  puts

  # Table header
  header = "  %-#{name_w}s" % "Reranker"
  q[:documents].each_with_index { |_, i| header += " | Doc #{i} score" }
  header += " | Ranking"
  puts header
  puts "  " + "-" * (header.length - 2)

  rerankers.each do |r|
    print "  Loading #{r[:name]}..."
    begin
      reranker = Candle::Reranker.from_pretrained(r[:id], device: device)
      print "\r" + " " * 60 + "\r"

      results = reranker.rerank(q[:query], q[:documents], apply_sigmoid: false)

      # Build a score-by-doc-index map
      score_by_idx = {}
      results.each { |res| score_by_idx[res[:doc_id]] = res[:score] }

      # Ranking order (doc indices sorted by score descending)
      ranking = results.map { |res| res[:doc_id] }.join(" > ")

      # Format row
      row = "  %-#{name_w}s" % r[:name]
      q[:documents].each_with_index do |_, i|
        score = score_by_idx[i]
        row += " | %11.4f  " % score
      end
      row += " | #{ranking}"
      puts row

      # Free memory
      reranker = nil
      GC.start(full_mark: true, immediate_sweep: true)
    rescue => e
      print "\r" + " " * 60 + "\r"
      puts "  %-#{name_w}s | ERROR: #{e.message[0..80]}" % r[:name]
    end
  end
  puts
end

# ============================================================
# Summary
# ============================================================
puts "=" * 120
puts "Architecture Summary"
puts "=" * 120
puts
puts "  %-#{name_w}s | %-14s | %-6s | %s" % ["Reranker", "Architecture", "Params", "Scoring Method"]
puts "  " + "-" * 90
rerankers.each do |r|
  method = r[:arch] == "Qwen3" ? "P(yes) from decoder logits" : "Cross-encoder logits"
  puts "  %-#{name_w}s | %-14s | %-6s | %s" % [r[:name], r[:arch], r[:params], method]
end
puts
