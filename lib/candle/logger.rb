require 'logger'

module Candle
  # Logging functionality for the Red Candle gem
  class << self
    # Get the current logger instance
    # @return [Logger] The logger instance
    def logger
      @logger ||= create_default_logger
    end

    # Set a custom logger instance
    # @param custom_logger [Logger] A custom logger instance
    def logger=(custom_logger)
      @logger = custom_logger
    end

    # Configure logging with a block
    # @yield [config] Configuration object
    def configure_logging
      config = LoggerConfig.new
      yield config if block_given?
      @logger = config.build_logger
    end

    private

    # Create the default logger with CLI-friendly settings
    # @return [Logger] Configured logger instance
    def create_default_logger
      logger = Logger.new($stderr)
      logger.level = default_log_level
      logger.formatter = cli_friendly_formatter
      logger
    end

    # Determine default log level based on environment variables
    # @return [Integer] Logger level constant
    def default_log_level
      # Support legacy CANDLE_VERBOSE for backward compatibility, but prefer explicit configuration
      return Logger::DEBUG if ENV['CANDLE_VERBOSE']
      Logger::WARN  # CLI-friendly: only show warnings/errors by default
    end

    # CLI-friendly formatter that outputs just the message
    # @return [Proc] Formatter proc
    def cli_friendly_formatter
      proc { |severity, datetime, progname, msg| "#{msg}\n" }
    end
  end

  # Configuration helper for logger setup
  class LoggerConfig
    attr_accessor :level, :output, :formatter

    def initialize
      @level = :warn
      @output = $stderr
      @formatter = :simple
    end

    # Build a logger from the configuration
    # @return [Logger] Configured logger
    def build_logger
      logger = Logger.new(@output)
      logger.level = normalize_level(@level)
      logger.formatter = build_formatter(@formatter)
      logger
    end

    # Set log level to debug (verbose output)
    def verbose!
      @level = :debug
    end

    # Set log level to info  
    def info!
      @level = :info
    end

    # Set log level to warn (default)
    def quiet!
      @level = :warn
    end

    # Set log level to error (minimal output)
    def silent!
      @level = :error
    end

    # Log to stdout instead of stderr
    def log_to_stdout!
      @output = $stdout
    end

    # Log to a file
    # @param file_path [String] Path to log file
    def log_to_file!(file_path)
      @output = file_path
    end

    # Disable logging completely
    def disable!
      @output = File::NULL
    end

    private

    # Convert symbol/string level to Logger constant
    # @param level [Symbol, String, Integer] Log level
    # @return [Integer] Logger level constant
    def normalize_level(level)
      case level.to_s.downcase
      when 'debug' then Logger::DEBUG
      when 'info' then Logger::INFO
      when 'warn', 'warning' then Logger::WARN
      when 'error' then Logger::ERROR
      when 'fatal' then Logger::FATAL
      else Logger::WARN
      end
    end

    # Build formatter based on type
    # @param formatter_type [Symbol] Type of formatter
    # @return [Proc] Formatter proc
    def build_formatter(formatter_type)
      case formatter_type
      when :simple, :cli
        proc { |severity, datetime, progname, msg| "#{msg}\n" }
      when :detailed
        proc do |severity, datetime, progname, msg|
          "[#{datetime.strftime('%Y-%m-%d %H:%M:%S')}] #{severity}: #{msg}\n"
        end
      when :json
        require 'json'
        proc do |severity, datetime, progname, msg|
          JSON.generate({
            timestamp: datetime.iso8601,
            level: severity,
            message: msg,
            program: progname
          }) + "\n"
        end
      else
        proc { |severity, datetime, progname, msg| "#{msg}\n" }
      end
    end
  end
end