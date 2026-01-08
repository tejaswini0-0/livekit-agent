import logging
from typing import Dict, Any, List
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    inference,
    MetricsCollectedEvent,
)
from livekit.plugins import noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# Change this to switch between different config files
ACTIVE_CONFIG = "agent_config.json"


class LatencyTracker:
    """Simple latency tracker for agent performance"""
    
    def __init__(self):
        self.eou_delays: List[float] = []
        self.llm_ttfts: List[float] = []
        self.tts_ttfbs: List[float] = []
        self.turn_count = 0
        
        self.current_eou = None
        self.current_llm = None
        self.current_tts = None
    
    def add_eou(self, delay_ms: float):
        self.current_eou = delay_ms
        logger.info(f"  End of Utterance: {delay_ms:.0f}ms")
        self._check_complete_turn()
    
    def add_llm(self, ttft_ms: float):
        self.current_llm = ttft_ms
        logger.info(f"  LLM (TTFT):       {ttft_ms:.0f}ms")
        self._check_complete_turn()
    
    def add_tts(self, ttfb_ms: float):
        self.current_tts = ttfb_ms
        logger.info(f"  TTS (TTFB):       {ttfb_ms:.0f}ms")
        self._check_complete_turn()
    
    def _check_complete_turn(self):
        if self.current_eou is not None and self.current_llm is not None and self.current_tts is not None:
            self.turn_count += 1
            total = self.current_eou + self.current_llm + self.current_tts
            
            self.eou_delays.append(self.current_eou)
            self.llm_ttfts.append(self.current_llm)
            self.tts_ttfbs.append(self.current_tts)
            
            logger.info(f"  {'-'*40}")
            logger.info(f"  TOTAL (Turn #{self.turn_count}): {total:.0f}ms\n")
            
            self.current_eou = None
            self.current_llm = None
            self.current_tts = None
    
    def print_summary(self):
        if self.turn_count == 0:
            logger.info("\nNo complete turns recorded")
            return
        
        totals = [e + l + t for e, l, t in zip(self.eou_delays, self.llm_ttfts, self.tts_ttfbs)]
        avg = sum(totals) / len(totals)
        sorted_totals = sorted(totals)
        p90_idx = int(len(sorted_totals) * 0.90)
        p90 = sorted_totals[p90_idx] if p90_idx < len(sorted_totals) else sorted_totals[-1]
        
        logger.info(f"\n{'='*50}")
        logger.info(f"LATENCY SUMMARY ({self.turn_count} turns)")
        logger.info(f"Average EOU: {sum(self.eou_delays)/len(self.eou_delays):.0f}ms")
        logger.info(f"Average LLM: {sum(self.llm_ttfts)/len(self.llm_ttfts):.0f}ms")
        logger.info(f"Average TTS: {sum(self.tts_ttfbs)/len(self.tts_ttfbs):.0f}ms")
        logger.info(f"Average Total: {avg:.0f}ms")
        logger.info(f"P90 Total: {p90:.0f}ms")
        logger.info(f"Min Total: {min(totals):.0f}ms")
        logger.info(f"Max Total: {max(totals):.0f}ms")
        logger.info(f"{'='*50}\n")


class AgentConfig:
    """Load and manage agent configuration from JSON file"""
    
    def __init__(self, config_path: str = "agent_config.json"):
        if not os.path.isabs(config_path):
            script_dir = Path(__file__).parent
            script_dir_config = script_dir / config_path
            if script_dir_config.exists():
                config_path = str(script_dir_config)
        
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                return self._get_default_config()
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "agent": {
                "name": "Assistant",
                "instructions": "You are a helpful assistant."
            },
            "models": {
                "llm": {"full_model_string": "openai/gpt-4o"},
                "stt": {"full_model_string": "deepgram/nova-3", "language": "en"},
                "tts": {
                    "full_model_string": "cartesia/sonic-3",
                    "voice": "79a125e8-cd45-4c13-8a67-188112f4dd22"
                }
            },
            "turn_detection": {"type": "multilingual", "enabled": True},
            "features": {"preemptive_generation": True, "noise_cancellation": True}
        }
    
    def get_agent_instructions(self) -> str:
        return self.config.get("agent", {}).get("instructions", "")
    
    def get_llm_config(self) -> Dict[str, Any]:
        return self.config.get("models", {}).get("llm", {})
    
    def get_stt_config(self) -> Dict[str, Any]:
        return self.config.get("models", {}).get("stt", {})
    
    def get_tts_config(self) -> Dict[str, Any]:
        return self.config.get("models", {}).get("tts", {})
    
    def get_turn_detection_config(self) -> Dict[str, Any]:
        return self.config.get("turn_detection", {})
    
    def get_features(self) -> Dict[str, Any]:
        return self.config.get("features", {})
    
    def is_preemptive_generation_enabled(self) -> bool:
        return self.get_features().get("preemptive_generation", True)
    
    def is_noise_cancellation_enabled(self) -> bool:
        return self.get_features().get("noise_cancellation", True)


def prewarm(proc: JobProcess):
    pass


async def entrypoint(ctx: JobContext):
    config_path = os.getenv("AGENT_CONFIG_PATH", ACTIVE_CONFIG)
    config = AgentConfig(config_path)

    llm_config = config.get_llm_config()
    stt_config = config.get_stt_config()
    tts_config = config.get_tts_config()
    turn_config = config.get_turn_detection_config()

    logger.info(f"Using config: {config_path}")

    session_kwargs = {
        "llm": inference.LLM(model=llm_config.get("full_model_string")),
        "stt": inference.STT(
            model=stt_config.get("full_model_string"),
            language=stt_config.get("language", "en")
        ),
        "tts": inference.TTS(
            model=tts_config.get("full_model_string"),
            voice=tts_config.get("voice", "")
        ),
        "preemptive_generation": config.is_preemptive_generation_enabled(),
    }
    
    if turn_config.get("enabled", True):
        turn_type = turn_config.get("type", "multilingual")
        if turn_type == "multilingual":
            session_kwargs["turn_detection"] = MultilingualModel()
    
    session = AgentSession(**session_kwargs)
    latency_tracker = LatencyTracker()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        m = ev.metrics
        
        try:
            # Check each metric type and add to tracker
            if hasattr(m, 'end_of_utterance_delay') and m.end_of_utterance_delay is not None:
                latency_tracker.add_eou(m.end_of_utterance_delay * 1000)
            
            if hasattr(m, 'ttft') and m.ttft is not None:
                latency_tracker.add_llm(m.ttft * 1000)
            
            if hasattr(m, 'ttfb') and m.ttfb is not None:
                latency_tracker.add_tts(m.ttfb * 1000)
        except Exception as e:
            pass

    async def log_summary():
        latency_tracker.print_summary()

    ctx.add_shutdown_callback(log_summary)

    room_input_kwargs = {}
    if config.is_noise_cancellation_enabled():
        room_input_kwargs["noise_cancellation"] = noise_cancellation.BVC()

    # Create generic agent with just instructions
    agent = Agent(instructions=config.get_agent_instructions())

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(**room_input_kwargs),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))