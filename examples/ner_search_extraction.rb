#!/usr/bin/env ruby
# frozen_string_literal: true

require_relative '../lib/candle'

# Smart Search Term Extractor
# Combines multiple approaches to extract "interesting" terms from text
class SmartSearchExtractor
  def initialize
    puts "Initializing Smart Search Extractor..."

    # 1. Standard NER for known entity types (PERSON, ORG, LOC, etc.)
    puts "  Loading NER model..."
    @ner = Candle::NER.from_pretrained("Babelscape/wikineural-multilingual-ner")

    # 2. Domain-specific pattern recognizers
    puts "  Setting up pattern recognizers..."

    # CAS Numbers (Chemical Abstracts Service)
    @cas_pattern = /\b\d{2,7}-\d{2}-\d\b/

    # Product/Compound codes
    @product_patterns = [
      /\b[A-Z]{2,4}-\d{3,5}\b/,     # Like "AB-12345"
      /\b[A-Z]{2,3}\d{3,4}[A-Z]?\b/, # Like "PD1" or "HER2"
      /\b[A-Z]+[-_]\d+[A-Z]*\b/      # General pattern
    ]

    # Clinical trial phases
    @trial_pattern = /\b(?:phase\s+)?(?:I{1,3}|[123])[ab]?\s+(?:clinical\s+)?trials?\b/i

    # 3. Noun phrase patterns for domain-specific terms
    @biomedical_patterns = [
      /\b(?:monoclonal|polyclonal|humanized|chimeric)?\s*antibod(?:y|ies)\b/i,
      /\b\w+\s+(?:inhibitor|agonist|antagonist|blocker|modulator)\b/i,
      /\b\w+\s+(?:receptor|enzyme|protein|kinase|ligand)\b/i,
      /\b(?:gene|protein|mRNA|DNA|RNA)\s+expression\b/i,
      /\b\w+\s+(?:pathway|mechanism|cascade|signaling)\b/i,
      /\b(?:biomarker|endpoint|efficacy|toxicity|safety)\s+\w+/i,
      /\b\w+\s+(?:assay|test|analysis|screening|validation)\b/i,
      /\b(?:therapeutic|diagnostic|prognostic)\s+\w+/i
    ]

    # 4. General interesting compound terms
    @compound_patterns = [
      /\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b/,  # Multi-word capitalized terms
      /\b[A-Z][a-z]+(?:-[A-Z][a-z]+)+\b/,    # Hyphenated terms
      /\b\w+[-]\d+\b/,                        # Terms with numbers
    ]

    # 5. Known important terms for the domain (gazetteer approach)
    @known_antibodies = %w[
      trastuzumab rituximab bevacizumab pembrolizumab nivolumab
      adalimumab infliximab cetuximab panitumumab ipilimumab
    ]

    @known_targets = %w[
      PD-1 PD-L1 CTLA-4 HER2 EGFR VEGF TNF CD20 CD19 BRAF
      ALK ROS1 NTRK MET RET FGFR IDH1 IDH2 FLT3 BCR-ABL
    ]

    @known_diseases = %w[
      cancer leukemia lymphoma melanoma carcinoma sarcoma
      myeloma glioma adenoma NSCLC SCLC CLL ALL AML CML
      diabetes alzheimer parkinson huntington COVID-19
    ]

    puts "  Initialization complete!"
  end

  def extract_all_terms(text)
    puts "\nAnalyzing text for search terms..."
    all_terms = {}

    # 1. Extract standard NER entities
    puts "  Running NER model..."
    begin
      entities = @ner.extract_entities(text)
      entities.each do |entity|
        term = entity["text"]
        all_terms[term] = {
          type: entity["label"],
          confidence: entity["confidence"] || 0.9,
          source: "NER",
          positions: [entity["start"]..entity["end"]]
        }
      end
      puts "    Found #{entities.length} named entities"
    rescue => e
      puts "    NER extraction skipped: #{e.message}"
    end

    # 2. Extract CAS numbers
    extract_patterns(text, @cas_pattern, "CAS_NUMBER", all_terms)

    # 3. Extract product codes
    @product_patterns.each do |pattern|
      extract_patterns(text, pattern, "PRODUCT_CODE", all_terms)
    end

    # 4. Extract clinical trial phases
    extract_patterns(text, @trial_pattern, "CLINICAL_TRIAL", all_terms)

    # 5. Extract biomedical noun phrases
    @biomedical_patterns.each do |pattern|
      extract_patterns(text, pattern, "BIOMEDICAL_TERM", all_terms, confidence: 0.8)
    end

    # 6. Extract general compound terms
    @compound_patterns.each do |pattern|
      extract_patterns(text, pattern, "COMPOUND_TERM", all_terms, confidence: 0.6)
    end

    # 7. Check for known terms (gazetteer)
    extract_known_terms(text, @known_antibodies, "ANTIBODY", all_terms)
    extract_known_terms(text, @known_targets, "DRUG_TARGET", all_terms)
    extract_known_terms(text, @known_diseases, "DISEASE", all_terms)

    # 8. Score and rank terms
    score_terms(all_terms)

    # Return sorted by score
    all_terms.sort_by { |_, info| -(info[:score] || 0) }.to_h
  end

  private

  def extract_patterns(text, pattern, label, terms_hash, confidence: 1.0)
    text.scan(pattern) do
      match = $~
      term = match[0]
      position = match.begin(0)..match.end(0)

      # Skip if already found with higher confidence
      next if terms_hash[term] && terms_hash[term][:confidence] > confidence

      terms_hash[term] ||= {
        type: label,
        confidence: confidence,
        source: "pattern",
        positions: []
      }
      terms_hash[term][:positions] << position
    end
  end

  def extract_known_terms(text, term_list, label, terms_hash)
    term_list.each do |term|
      regex = /\b#{Regexp.escape(term)}\b/i
      if text.match?(regex)
        terms_hash[term] ||= {
          type: label,
          confidence: 1.0,
          source: "gazetteer",
          positions: []
        }

        # Find all positions
        text.scan(regex) do
          match = $~
          terms_hash[term][:positions] << (match.begin(0)..match.end(0))
        end
      end
    end
  end

  def score_terms(terms_hash)
    terms_hash.each do |term, info|
      next unless term  # Skip nil terms

      # Base score from confidence
      score = info[:confidence] || 0.5

      # Boost for specific types
      case info[:type]
      when "ANTIBODY", "DRUG_TARGET", "CAS_NUMBER"
        score *= 1.5
      when "PRODUCT_CODE", "CLINICAL_TRIAL"
        score *= 1.3
      when "ORG", "BIOMEDICAL_TERM"
        score *= 1.2
      when "PERSON", "LOC"
        score *= 0.8  # Less relevant for biomedical search
      end

      # Boost for known terms
      score *= 1.2 if info[:source] == "gazetteer"

      # Boost for multiple occurrences
      if info[:positions] && info[:positions].length > 1
        occurrences = info[:positions].length
        score *= (1 + Math.log(occurrences) * 0.1)
      end

      # Boost for term length (longer terms are often more specific)
      score *= (1 + term.to_s.split(/\s+/).length * 0.05)

      info[:score] = score
    end
  end

  def format_results(terms_hash)
    puts "\n" + "="*60
    puts "EXTRACTED SEARCH TERMS"
    puts "="*60

    terms_hash.each_with_index do |(term, info), index|
      next if term.nil? || term.empty?  # Skip nil or empty terms

      puts "\n#{index + 1}. \"#{term}\""
      puts "   Type: #{info[:type]}"
      puts "   Confidence: #{'%.2f' % (info[:confidence] || 0)}"
      puts "   Score: #{'%.2f' % (info[:score] || 0)}"
      puts "   Source: #{info[:source]}"
      puts "   Occurrences: #{info[:positions]&.length || 0}"
    end

    puts "\n" + "="*60
    puts "SEARCH QUERY SUGGESTIONS"
    puts "="*60

    # Group by type for structured search
    by_type = terms_hash.group_by { |_, info| info[:type] }

    # Suggest high-precision searches
    high_precision = terms_hash.select { |_, info| info[:confidence] >= 0.9 }
    if high_precision.any?
      puts "\nHigh-precision terms (confidence >= 0.9):"
      puts "  " + high_precision.keys.first(5).join(" AND ")
    end

    # Suggest entity-specific searches
    if by_type["ANTIBODY"] || by_type["DRUG_TARGET"]
      antibodies = (by_type["ANTIBODY"] || []).map(&:first)
      targets = (by_type["DRUG_TARGET"] || []).map(&:first)
      drugs = antibodies + targets
      puts "\nDrug/Target search:"
      puts "  " + drugs.first(3).join(" OR ")
    end

    # Suggest clinical search
    if by_type["CLINICAL_TRIAL"] || by_type["DISEASE"]
      trials = (by_type["CLINICAL_TRIAL"] || []).map(&:first)
      diseases = (by_type["DISEASE"] || []).map(&:first)
      clinical = trials + diseases
      puts "\nClinical search:"
      puts "  " + clinical.first(3).join(" AND ")
    end

    # Top weighted terms for general search
    top_terms = terms_hash.sort_by { |_, info| -(info[:score] || 0) }.first(5)
    if top_terms.any?
      puts "\nTop weighted terms for broad search:"
      puts "  " + top_terms.map(&:first).join(" ")
    end

    puts "\n" + "="*60
  end
end

# MAIN EXECUTION
if __FILE__ == $0
  # Sample biomedical text
  text = <<~TEXT
    We are conducting a Phase II clinical trial for advanced melanoma patients using
    pembrolizumab (MK-3475), a humanized monoclonal antibody targeting PD-1. The study
    will compare the efficacy of pembrolizumab monotherapy versus combination therapy
    with ipilimumab, a CTLA-4 inhibitor.

    Previous studies with compound AB-12345 (CAS: 1234567-89-0) showed promising results
    in BRAF-mutant melanoma. The primary endpoint is progression-free survival, with
    secondary endpoints including overall response rate and safety profile.

    Dr. Sarah Johnson from Memorial Sloan Kettering Cancer Center is the principal
    investigator. The trial is registered as NCT-04567890 and is recruiting patients
    at multiple sites across the United States, including locations in New York,
    California, and Texas.

    Biomarker analysis will include PD-L1 expression testing, tumor mutational burden
    assessment, and gene expression profiling. The HER2 receptor status will also be
    evaluated for potential stratification.

    This innovative approach combines checkpoint inhibitor therapy with targeted
    kinase inhibitors to overcome resistance mechanisms in advanced melanoma.
  TEXT

  puts "Red Candle - Smart Search Term Extraction Demo"
  puts "="*60
  puts "\nINPUT TEXT:"
  puts "-"*40
  puts text
  puts "-"*40

  # Create extractor and analyze text
  extractor = SmartSearchExtractor.new
  terms = extractor.extract_all_terms(text)

  # Display results
  extractor.send(:format_results, terms)

  puts "\nTOTAL TERMS EXTRACTED: #{terms.length}"
  puts "\nExample parallel search code:"
  puts "-"*40
  puts <<~CODE
    # You could now search in parallel:
    results = terms.first(10).map do |term, info|
      Thread.new do
        weight = info[:score]
        # search_results = search_database(term)  # Your search function
        { term: term, weight: weight }
      end
    end.map(&:value)

    # Then rerank combined results using Candle::Reranker
    # reranker = Candle::Reranker.from_pretrained("BAAI/bge-reranker-base")
    # final_results = reranker.rerank(query, all_results)
  CODE
end