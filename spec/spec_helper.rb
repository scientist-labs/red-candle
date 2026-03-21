# SimpleCov must be started before requiring the code to track
require 'simplecov'
SimpleCov.start do
  add_filter '/spec/'
  add_filter '/test/'
  add_filter '/vendor/'
  add_filter '/ext/'  # Native extensions can't be tracked
  track_files 'lib/**/*.rb'
  
  # Add groups for better organization
  add_group 'Models', 'lib/candle'
  add_group 'Core', 'lib/candle.rb'
end

require 'bundler/setup'
Bundler.require(:default)

# Load the library
require 'candle'

# Load support files
Dir[File.join(__dir__, 'support', '**', '*.rb')].each { |f| require f }

RSpec.configure do |config|
  # Enable flags like --only-failures and --next-failure
  config.example_status_persistence_file_path = ".rspec_status"

  # Disable monkey patching
  config.disable_monkey_patching!

  # Use expect syntax only
  config.expect_with :rspec do |c|
    c.syntax = :expect
  end

  # Random order with seed
  config.order = :random
  Kernel.srand config.seed

  # Filter examples by tags
  config.filter_run_when_matching :focus
  
  # Run specs in random order to surface order dependencies
  config.order = :random
  
  # Print the slowest examples and example groups
  config.profile_examples = 10 if ENV['PROFILE_SPECS']

  # Include helpers
  config.include DeviceHelpers
  
  # Configure clean test output
  config.before(:suite) do
    $VERBOSE = nil
    
    # Set logger to error-only during tests (clean test output)
    Candle.configure_logging { |c| c.silent! }
  end
  
  # Print test environment info (use --verbose or -v for details)
  config.before(:suite) do
    if ARGV.include?('--verbose') || ARGV.include?('-v')
      puts "\nCandle RSpec Test Environment:"
      puts "  Ruby version: #{RUBY_VERSION}"
      puts "  Available devices: #{DeviceHelpers.available_devices.join(', ')}"
      puts "  Testing devices: #{DeviceHelpers.devices_to_test.join(', ')}"
      puts
      
      # Also enable verbose logging for tests
      Candle.configure_logging { |c| c.verbose! }
    end
  end
  
  # Global cleanup after suite completes
  config.after(:suite) do
    # Clear all cached models
    ModelCache.clear! if defined?(ModelCache)
    
    # Force garbage collection to clean up native resources
    GC.start(full_mark: true, immediate_sweep: true)
    
    # Give the system a moment to clean up
    sleep(0.1)
  end
  
  # Aggressive cleanup after each top-level describe block to free native memory.
  # Native (Rust/Candle) models loaded via from_pretrained hold large mmap'd
  # tensors. Ruby's GC won't collect them while instance variables still hold
  # references, so we explicitly nil any model-holding ivars before GC.
  config.after(:all) do |example_group|
    if example_group.class.top_level?
      # Nil all instance variables that might hold native model objects
      instance_variables.each do |ivar|
        val = instance_variable_get(ivar)
        if val.is_a?(Candle::LLM) || val.is_a?(Candle::EmbeddingModel) ||
           val.is_a?(Candle::Reranker) || val.is_a?(Candle::NER) ||
           val.is_a?(Candle::Tokenizer)
          instance_variable_set(ivar, nil)
        end
      end

      # Clear any ModelCache entries
      ModelCache.clear! if defined?(ModelCache)

      # Aggressive GC to reclaim native memory
      GC.start(full_mark: true, immediate_sweep: true)
    end
  end
end