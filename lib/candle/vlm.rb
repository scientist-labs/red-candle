# frozen_string_literal: true

module Candle
  class VLM
    class << self
      def from_pretrained(model_id, device: nil, **options)
        device_obj = case device
                     when "cpu" then Candle::Device.cpu
                     when "metal" then Candle::Device.metal
                     when "cuda" then Candle::Device.cuda
                     when Candle::Device then device
                     when nil then nil
                     else Candle::Device.best
                     end
        _create(model_id, device_obj)
      end
    end

    def describe(image_path, max_length: 256)
      _describe(image_path, max_length)
    end

    def ask(image_path, question, max_length: 256)
      _ask(image_path, question, max_length)
    end

    def inspect
      "#<Candle::VLM model_id=#{model_id.inspect} device=#{device}>"
    end
  end
end
