import logging
from typing import List, Optional
import statistics
import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    inference,
    metrics,
    function_tool,
    RunContext,
)
from livekit.plugins import noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class SimpleLatencyTracker:
    """Simple tracker that collects latency from LiveKit metrics"""
    
    def __init__(self):
        self.eou_delays: List[float] = []
        self.llm_ttfts: List[float] = []
        self.tts_ttfbs: List[float] = []
        self.turn_count = 0
        
        # Temporary storage for current turn metrics
        self.current_eou: Optional[float] = None
        self.current_llm: Optional[float] = None
        self.current_tts: Optional[float] = None
    
    def add_eou(self, delay_ms: float):
        """Add EOU delay for current turn"""
        self.current_eou = delay_ms
        logger.info(f"  End of Utterance: {delay_ms:.0f}ms")
        self._check_complete_turn()
    
    def add_llm(self, ttft_ms: float):
        """Add LLM TTFT for current turn"""
        self.current_llm = ttft_ms
        logger.info(f"  LLM (TTFT):        {ttft_ms:.0f}ms")
        self._check_complete_turn()
    
    def add_tts(self, ttfb_ms: float):
        """Add TTS TTFB for current turn"""
        self.current_tts = ttfb_ms
        logger.info(f"  TTS (TTFB):        {ttfb_ms:.0f}ms")
        self._check_complete_turn()
    
    def _check_complete_turn(self):
        """Check if we have all metrics for a complete turn"""
        if self.current_eou is not None and self.current_llm is not None and self.current_tts is not None:
            self.turn_count += 1
            total = self.current_eou + self.current_llm + self.current_tts
            
            self.eou_delays.append(self.current_eou)
            self.llm_ttfts.append(self.current_llm)
            self.tts_ttfbs.append(self.current_tts)
            
            logger.info(f"  {'-'*60}")
            logger.info(f"  TOTAL (Turn #{self.turn_count}): {total:.0f}ms\n")
            
            # Reset for next turn
            self.current_eou = None
            self.current_llm = None
            self.current_tts = None
    
    def print_summary(self):
        """Print final summary statistics"""
        if self.turn_count == 0:
            logger.info("\nNo complete turns recorded")
            return
        
        def calc_stats(values: List[float], name: str):
            if not values:
                return
            
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            
            avg = statistics.mean(values)
            p50 = sorted_vals[int(n * 0.50)] if n > 0 else 0
            p90 = sorted_vals[int(n * 0.90)] if n > 1 else sorted_vals[0]
            p99 = sorted_vals[int(n * 0.99)] if n > 1 else sorted_vals[0]
            
            logger.info(f"\n{name}:")
            logger.info(f"  Average: {avg:.0f}ms")
            logger.info(f"  P50:     {p50:.0f}ms")
            logger.info(f"  P90:     {p90:.0f}ms")
            logger.info(f"  P99:     {p99:.0f}ms")
            logger.info(f"  Min:     {min(values):.0f}ms")
            logger.info(f"  Max:     {max(values):.0f}ms")
        
        # Calculate total latencies
        totals = [e + l + t for e, l, t in zip(self.eou_delays, self.llm_ttfts, self.tts_ttfbs)]
        
        logger.info("\n" + "="*70)
        logger.info("MULTILINGUAL TURN DETECTION + LATENCY RESULTS")
        logger.info("="*70)
        logger.info(f"Total Turns: {self.turn_count}")
        
        calc_stats(self.eou_delays, "End of Utterance (MultilingualModel)")
        calc_stats(self.llm_ttfts, "LLM (OpenAI GPT-4o)")
        calc_stats(self.tts_ttfbs, "TTS (Cartesia Sonic-3)")
        calc_stats(totals, "TOTAL LATENCY")
        
        # Performance assessment
        total_p90 = sorted(totals)[int(len(totals) * 0.90)] if len(totals) > 1 else totals[0]
        total_p99 = sorted(totals)[int(len(totals) * 0.99)] if len(totals) > 1 else totals[0]
        
        logger.info(f"\nP90 Total Latency: {total_p90:.0f}ms")
        logger.info(f"P99 Total Latency: {total_p99:.0f}ms")
        
        if total_p90 < 500:
            logger.info("Status: EXCELLENT - Sub-500ms")
        elif total_p90 < 800:
            logger.info("Status: GOOD - Sub-800ms")
        else:
            logger.info("Status: Needs further optimization")
        
        logger.info("="*70 + "\n")


class CaliforniaBurritoAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Maria, a friendly and enthusiastic customer service agent for California Burrito, a popular Mexican food restaurant chain in India.
            
            ABOUT CALIFORNIA BURRITO:
            California Burrito is India's favorite destination for authentic Mexican food with a delicious twist. 
            We serve fresh, made-to-order burritos, tacos, quesadillas, nachos, and more with bold flavors that Indians love.
            Founded in 2016, we now have multiple locations across major Indian cities bringing authentic Mexican street food culture to India.
            
            OUR SIGNATURE DISHES:
            - Habanero Burrito: Our signature burrito loaded with your choice of protein, rice, beans, cheese, sour cream, and fresh veggies
            - Chicken Burrito: A fusion favorite combining Mexican and Indian flavors
            - Veggie Supreme Burrito: Packed with grilled vegetables, black beans, and guacamole
            - Crunchy Tacos: Three crispy tacos with seasoned filling and fresh toppings
            - Loaded Nachos: Crispy tortilla chips topped with cheese, jalapeños, salsa, and your choice of protein
            - Quesadillas: Grilled flour tortillas stuffed with melted cheese and fillings
            - Burrito Bowls: All the goodness of a burrito in a bowl, perfect for low-carb diets
            - Churros: Sweet cinnamon-sugar coated dessert sticks
            
            PROTEIN OPTIONS:
            - Grilled Chicken (most popular)
            - Chicken Tikka (Indian fusion)
            - Paneer Tikka (vegetarian favorite)
            - Black Beans (vegan option)
            - Grilled Vegetables
            
            LOCATIONS IN INDIA:
            - Bangalore: Indiranagar, Koramangala, Whitefield, HSR Layout
            - Mumbai: Bandra, Powai, Andheri
            - Pune: Koregaon Park, Viman Nagar
            - Hyderabad: Banjara Hills, Hitech City
            - Delhi NCR: Cyber Hub Gurgaon, Connaught Place, Noida
            
            HOURS:
            - Monday to Sunday: 11 AM to 11 PM
            - Kitchen closes at 10:30 PM
            
            ORDERING OPTIONS:
            - Dine-in at any of our locations
            - Online ordering through our website and app
            - Food delivery via Swiggy, Zomato, and our own delivery service
            - Catering for parties and events (minimum order 10 people)
            
            PRICING:
            - Burritos: Rs 250 - Rs 350
            - Tacos (3 piece): Rs 200 - Rs 280
            - Quesadillas: Rs 220 - Rs 300
            - Nachos: Rs 180 - Rs 250
            - Bowls: Rs 280 - Rs 380
            - Combo Meals: Rs 350 - Rs 450 (includes main + side + drink)
            - Churros: Rs 120
            
            SPECIAL OFFERS:
            - Taco Tuesday: Buy 2 get 1 free on all tacos every Tuesday
            - Weekend Combo: Rs 399 for burrito + nachos + drink (Saturday & Sunday)
            - Student Discount: 15% off with valid student ID (lunch hours only)
            - Loyalty Program: Earn 1 point per Rs 100 spent, redeem for free food
            
            YOUR ROLE:
            - Greet customers warmly and introduce yourself as Maria from California Burrito
            - Help customers with menu recommendations based on their preferences
            - Take orders for pickup or delivery
            - Answer questions about ingredients, allergens, and customization
            - Provide information about locations and hours
            - Handle special requests for catering and large orders
            - Share current promotions and deals
            - Resolve complaints with empathy and offer solutions
            
            COMMUNICATION STYLE:
            - Be warm, enthusiastic, and genuinely helpful
            - Keep responses conversational and friendly (2-3 sentences)
            - Use food descriptions that make dishes sound delicious
            - Show excitement about Mexican food and our menu
            - Always offer to help with anything else they need
            - Never use emojis, asterisks, or special formatting in speech
            - If someone asks for something not on the menu, politely suggest alternatives
            
            IMPORTANT GUIDELINES:
            - Always ask about dietary restrictions or allergies when taking orders
            - Confirm order details including location for delivery
            - For catering, connect them with the manager for special arrangements
            - Be understanding if delivery is delayed and offer apologies
            - Always mention our loyalty program to new customers""",
        )
        
        self.current_order = {}
        self.customer_location = ""

    @function_tool
    async def check_menu_item(self, context: RunContext, item_name: str):
        """Get detailed information about a specific menu item."""
        logger.info(f"Checking menu item: {item_name}")
        
        menu = {
            "habanero burrito": "Our signature California Burrito is Rs 320 and comes with your choice of protein, Mexican rice, refried beans, melted cheese, sour cream, fresh lettuce, pico de gallo, and our special house sauce. It's huge and incredibly satisfying. Would you like to customize it?",
            "chicken burrito": "The Chicken Tikka Burrito at Rs 340 is our fusion masterpiece. Marinated tikka chicken pieces with rice, beans, cheese, onions, and a tangy mint chutney wrapped in a warm tortilla. It's the perfect blend of Mexican and Indian flavors.",
            "tacos": "We serve three crispy tacos for Rs 250 with your choice of filling, lettuce, cheese, salsa, and sour cream. They're crunchy, fresh, and absolutely delicious. What protein would you prefer?",
            "nachos": "Our Loaded Nachos are Rs 220 and come with crispy tortilla chips covered in melted cheese, jalapeños, black beans, salsa, guacamole, and sour cream. You can add chicken or beef for Rs 50 extra.",
            "quesadilla": "Quesadillas start at Rs 250 and feature a grilled flour tortilla stuffed with melted cheese and your choice of filling. Served with salsa and sour cream on the side. Perfect for a lighter meal.",
            "burrito bowl": "The Burrito Bowl gives you all the burrito goodness without the tortilla. Starting at Rs 300, it includes rice, beans, protein, cheese, veggies, and all the toppings. Great for a healthier option.",
            "churros": "Our Churros are Rs 120 for four pieces. Fresh fried dough coated in cinnamon sugar, served with chocolate dipping sauce. The perfect sweet ending to your meal.",
        }
        
        item_lower = item_name.lower()
        for key in menu:
            if key in item_lower or item_lower in key:
                return menu[key]
        
        return f"I don't see {item_name} on our menu, but we have amazing burritos, tacos, quesadillas, nachos, and bowls. What type of Mexican food are you craving and I can suggest something delicious?"


def prewarm(proc: JobProcess):
    # No prewarm needed with MultilingualModel only
    pass


async def entrypoint(ctx: JobContext):
    latency_tracker = SimpleLatencyTracker()
    
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    logger.info("Using OpenAI GPT-4o for LLM inference")
    logger.info("Using MultilingualModel for context-aware turn detection")

    session = AgentSession(
        # OpenAI GPT-4o
        llm=inference.LLM(model="openai/gpt-4o"),
        
        # Deepgram Nova-3 for fastest STT
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        
        # Cartesia Sonic-3 for fast TTS
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="79a125e8-cd45-4c13-8a67-188112f4dd22",
        ),
        
        # MultilingualModel - Context-aware turn detection
        # Analyzes transcript content to detect natural conversation turns
        turn_detection=MultilingualModel(),
        
        # Start generating before user finishes
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
        
        m = ev.metrics
        metric_type = type(m).__name__
        
        try:
            if metric_type == "EOUMetrics":
                if hasattr(m, 'end_of_utterance_delay') and m.end_of_utterance_delay is not None:
                    latency_tracker.add_eou(m.end_of_utterance_delay * 1000)
            
            elif metric_type == "LLMMetrics":
                if hasattr(m, 'ttft') and m.ttft is not None:
                    latency_tracker.add_llm(m.ttft * 1000)
            
            elif metric_type == "TTSMetrics":
                if hasattr(m, 'ttfb') and m.ttfb is not None:
                    latency_tracker.add_tts(m.ttfb * 1000)
        
        except Exception as e:
            pass

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
        latency_tracker.print_summary()

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=CaliforniaBurritoAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))