module Candle
  module BuildInfo
    def self.display_cuda_info
      info = Candle.build_info
      
      # CUDA info is now controlled by logger level
      
      if info["cuda_available"] == false
        # :nocov:
        # Check if CUDA could be available on the system
        cuda_potentially_available = ENV['CUDA_ROOT'] || ENV['CUDA_PATH'] || 
                                   File.exist?('/usr/local/cuda') || File.exist?('/opt/cuda')
        
        if cuda_potentially_available
          Candle.logger.warn "=" * 80
          Candle.logger.warn "Red Candle: CUDA detected on system but not enabled in build."
          Candle.logger.warn "This may be due to CANDLE_DISABLE_CUDA being set during installation."
          Candle.logger.warn "To enable CUDA support, reinstall without CANDLE_DISABLE_CUDA set."
          Candle.logger.warn "=" * 80
        end
        # :nocov:
      end
    end
    
    def self.cuda_available?
      Candle.build_info["cuda_available"]
    end

    def self.metal_available?
      Candle.build_info["metal_available"]
    end

    def self.mkl_available?
      Candle.build_info["mkl_available"]
    end

    def self.accelerate_available?
      Candle.build_info["accelerate_available"]
    end

    def self.cudnn_available?
      Candle.build_info["cudnn_available"]
    end

    def self.summary
      info = Candle.build_info
      
      available_backends = []
      available_backends << "Metal" if info["metal_available"]
      available_backends << "CUDA" if info["cuda_available"]
      available_backends << "CPU"
      
      {
        default_device: info["default_device"],
        available_backends: available_backends,
        cuda_available: info["cuda_available"],
        metal_available: info["metal_available"],
        mkl_available: info["mkl_available"],
        accelerate_available: info["accelerate_available"],
        cudnn_available: info["cudnn_available"]
      }
    end
  end
end

# Display CUDA info on load
Candle::BuildInfo.display_cuda_info