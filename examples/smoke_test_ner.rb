# frozen_string_literal: true

require_relative "smoke_test_helper"

section "NER (Model-based)"

ner = test("load") do
  Candle::NER.from_pretrained("dslim/bert-base-NER", device: $device, tokenizer: "bert-base-cased")
end

if ner
  text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
  test("extract_entities") { ner.extract_entities(text) }
  test("entity_types") { ner.entity_types }
  test("analyze") { ner.analyze(text) }
  test("format_entities") { ner.format_entities(text) }
  test("extract_entity_type") { ner.extract_entity_type(text, "PER") }
  test("supports_entity?") { ner.supports_entity?("PER") }
  test("inspect") { ner.inspect }
end

section "NER (Pattern & Gazetteer)"

test("PatternEntityRecognizer") do
  rec = Candle::PatternEntityRecognizer.new("EMAIL", [/\b[\w.]+@[\w.]+\.\w+\b/])
  results = rec.recognize("Contact us at hello@example.com")
  raise "no matches" if results.empty?
  results
end

test("GazetteerEntityRecognizer") do
  rec = Candle::GazetteerEntityRecognizer.new("LANG", ["Ruby", "Python", "Rust"])
  results = rec.recognize("I love Ruby and Rust")
  raise "no matches" if results.empty?
  results
end

test("HybridNER") do
  hybrid = Candle::HybridNER.new
  hybrid.add_pattern_recognizer("EMAIL", [/\b[\w.]+@[\w.]+\.\w+\b/])
  hybrid.add_gazetteer_recognizer("LANG", ["Ruby", "Python"])
  results = hybrid.extract_entities("Email ruby@dev.com about Ruby")
  raise "no matches" if results.empty?
  results
end

smoke_test_summary
