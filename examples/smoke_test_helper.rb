# frozen_string_literal: true

require "candle"

$device = Candle::Device.best
$passed = []
$failed = []

def test(name)
  print "  #{name}... "
  result = yield
  puts "✅"
  $passed << name
  result
rescue => e
  puts "❌ #{e.message[0..150]}"
  $failed << { name: name, error: e.message }
  nil
end

def smoke_test_summary
  puts
  puts "=" * 80
  puts "  ✅ Passed: #{$passed.length}"
  if $failed.any?
    puts "  ❌ Failed: #{$failed.length}"
    $failed.each { |f| puts "     - #{f[:name]}: #{f[:error][0..100]}" }
  end
  puts "  Total: #{$passed.length + $failed.length}"
  puts "=" * 80
  exit($failed.empty? ? 0 : 1)
end

def section(title)
  puts "=" * 80
  puts title
  puts "-" * 80
end
