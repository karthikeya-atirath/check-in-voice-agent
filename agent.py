import logging
import os
import sys
import json
import httpx
import asyncio

from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.beta.workflows import TaskGroup, TaskCompletedEvent
from livekit import agents
from livekit.agents import Agent, AgentTask, AgentSession, AgentServer, RunContext, StopResponse, inference, function_tool, room_io, JobContext, TurnHandlingOptions
from livekit.agents.job import get_job_context
from livekit.plugins import silero, ai_coustics, cartesia, deepgram
from pypdf import PdfReader
from llm_service import get_llm

load_dotenv()

server = AgentServer()

def load_customer():
    with open("customer 1.json", "r") as file:
        return json.load(file)

customer_data = load_customer()

query_category_value = ""
priority_level_value = 1
callback_required_value = False
closing_value = ""
report_saved = False
completed_tasks_value: list[str] = []
IST = timezone(timedelta(hours=5, minutes=30))
started_at = datetime.now(IST).isoformat()
report_filename = f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def find_customer_trip_pdf() -> Path | None:
    configured_path = os.getenv("CUSTOMER_TRIP_PDF")
    if configured_path:
        path = Path(configured_path.strip().strip('"'))
        if path.exists():
            return path

        project_path = Path.cwd() / path
        if project_path.exists():
            return project_path

    project_pdfs = sorted(Path.cwd().glob("*.pdf"))
    return project_pdfs[0] if project_pdfs else None

@lru_cache(maxsize=1)
def load_customer_trip_pdf_text() -> dict:
    pdf_path = find_customer_trip_pdf()
    if not pdf_path:
        return {
            "source_file": "",
            "text": "",
            "error": f"No PDF found in {Path.cwd()}",
        }

    try:
        pdf_bytes = pdf_path.read_bytes()
        if not pdf_bytes.rstrip().endswith(b"%%EOF"):
            pdf_bytes = pdf_bytes.rstrip() + b"\n%%EOF\n"

        reader = PdfReader(BytesIO(pdf_bytes))
        page_texts = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                page_texts.append(f"[Page {page_number}]\n{text.strip()}")

        return {
            "source_file": str(pdf_path),
            "page_count": len(reader.pages),
            "text": "\n\n".join(page_texts),
            "error": "",
        }
    except Exception as e:
        return {
            "source_file": str(pdf_path),
            "text": "",
            "error": f"Could not read trip PDF: {str(e)}",
        }

def get_customer_trip_pdf_prompt_context() -> str:
    pdf_data = load_customer_trip_pdf_text()
    if pdf_data.get("error"):
        return pdf_data["error"]

    return (
        f"Source file: {pdf_data.get('source_file')}\n"
        f"Pages: {pdf_data.get('page_count')}\n\n"
        f"{pdf_data.get('text', '')}"
    )

@function_tool
async def weather(location: str):
    "Use this tool when the call flow or user asks about weather in an area."

    try:
        response = httpx.get(
            "https://api.weatherstack.com/current",
            params = {"access_key": "95ffa171c26da294a5f8e2e8f2c9e294", "query": location}
        )
        data = response.json()

        if "error" in data:
            return f"Weather unavailable: {data['error']['info']}"

        return {
            "location": data["location"]["name"],
            "temperature": data["current"]["temperature"],
            "humidity": data["current"]["humidity"],
            "weather": data["current"]["weather_descriptions"][0],
            "wind_speed": data["current"]["wind_speed"]
        }
    
    except Exception as e:
        return f"Weather tool failed: {str(e)}"

@function_tool
async def answer_faqs(destination: str):
    """ Use this tool whenever the customer asks any destination related FAQ, local rules, Baggage allowance, travel tips, Pickup coordination,
    food, Emergency contacts, currency, best time to visit, or general destination question. """
    try:
        response = httpx.post(
            "http://35.200.238.161:8000/v1/get_destination_details",
            json = {"search_term": destination},
            timeout=15
        )

        response.raise_for_status()

    except httpx.HTTPError as e:
        error_detail = response.text if "response" in locals() else str(e)
        return {"error": f"FAQ API failed: {str(e)}", "detail": error_detail}
    
    try:
        faq_data = response.json()
    
    except ValueError as e:
        return {"error": f"FAQ API returned invalid json: {str(e)}"}

    return faq_data

@function_tool
async def get_package_data(packageid: str):
    "Use this to retrieve package details for a given package ID."

    try:
        response = httpx.post(
            "http://35.200.238.161:8000/v1/get_raw_response",
            json={"packageId": packageid},
            timeout=15,
        )
        response.raise_for_status()

    except httpx.HTTPError as e:
        return {"error": f"Package data tool failed: {str(e)}"}

    class_id = str(customer_data.get("packageClassId", ""))

    try:
        package_data = response.json()

    except ValueError as e:
        return {"error": f"Package data tool returned invalid json: {str(e)}"}

    package_specific = package_data.get("raw_response", {}).get("packageDetail", {})

    policies = [
        policy
        for policy in package_specific.get("tcilHolidayTermsConditionsCollection", [])
        if str(policy.get("packageClassId")) == class_id
    ]

    accommodations = [
        item
        for item in package_specific.get("tcilHolidayAccomodationCollection", [])
        if str(item.get("packageClassId")) == class_id
    ]

    flights = [
        item
        for item in package_specific.get("tcilHolidayFlightCollection", [])
        if str(item.get("packageClassId")) == class_id
    ]

    prices = [
        item
        for item in package_specific.get("tcilHolidayLtPricingCollection", [])
        if str(item.get("packageClassId")) == class_id
    ]


    cities = package_specific.get("tcilHolidayCityCollection", [])
    inclusions_exclusions = package_specific.get("tcilHolidayIncludeExcludeCollection", [])
    itinerary = package_specific.get("tcilHolidayItineraryCollection", [])

    return {
        "packageClassId": class_id,
        "policies": policies,
        "accommodations": accommodations,
        "cities": cities,
        "inclusions_exclusions": inclusions_exclusions,
        "itinerary": itinerary,
        "flights": flights,
        "prices": prices,
    }

@function_tool
async def retrieve_customer_trip_pdf(question: str = ""):
    """
    Use this before answering any customer-specific booking question.
    The question parameter must contain the exact customer question.
    Return only the most relevant PDF text if possible.
    """
    pdf_data = load_customer_trip_pdf_text()
    if pdf_data.get("error"):
        return {"error": pdf_data["error"]}

    return {
        "question": question,
        "source_file": pdf_data.get("source_file"),
        "page_count": pdf_data.get("page_count"),
        "full_pdf_text": pdf_data.get("text", ""),
    }

@function_tool
async def call_details(
    query_category: str = "",
    priority_level: int = 0,
    callback_required: str = "",
    closing: str = "",
):
    """Store report fields. query_category must be one of:
    no_query, trip_confirmation, itinerary, accommodation, transport,
    payment, documents, baggage_meals, weather_packing, destination_info,
    wrong_contact, cancellation_or_reschedule, support_callback.
    """

    global callback_required_value, closing_value, priority_level_value, query_category_value

    if query_category:
        query_category_value = query_category

    if priority_level:
        priority_level_value = priority_level

    if callback_required:
        callback_required_value = callback_required.strip().lower() in (
            "yes",
            "true",
            "required",
        )

    if closing:
        closing_value = closing

    return {"status": "saved"}

class InterruptibleTask(AgentTask[str]):
    """
    Base task that:
    1. Speaks
    2. If interrupted, answers user
    3. Resumes only the missing part of the original flow
    """

    def __init__(self, *, instructions: str, customer: dict | None = None) -> None:
        c = customer or customer_data
        self.customer = c
        self._pending_resume_lines: list[str] = []
        self._awaiting_clarification_confirmation = False
        self._lock = asyncio.Lock()

        shared_instructions = f"""
        Speak like a helpful, polite human on a phone call. Keep replies short.
        Use courteous phrasing naturally, including "please" and "thank you", without overdoing it.

        When the customer directly answers your pending question, briefly acknowledge it first with a short natural word like "Okay", "Alright", or "Got it" before moving on. Vary the word each time, do not reuse the same one two turns in a row.

        When the customer does not answer your pending question, for example they go off-topic, give an unrelated or unclear reply, or stay silent, stay calm and patient. Do not sound impatient, repetitive in tone, or scripted. Gently and politely rephrase or repeat the question once, for example "No worries, just to confirm, ..." or "Take your time, I just need to know ...", and wait for the answer. Never sound annoyed, terse, or robotic when the customer needs the question repeated.
        If the customer is silent or unclear, use one gentle filler or clarifying question and wait.
        If the customer interrupts or asks a question, answer it first, then return to the current pending question.
        If you previously asked "Does that answer your question?" after an interruption, and the customer now confirms with yes, okay, clear, got it, or similar, call resume_script to continue the paused script. If the customer says no, not clear, or asks something else, clarify further and ask "Does that answer your question?" again, then wait.
        Do not call end_call inside this task.

        Source rules:
        Use retrieve_customer_trip_pdf first for customer-specific booking details such as reporting time, meeting point, pickup, assigned hotel, tickets, vouchers, visa, passport, payment status, total cost, paid amount, or pending amount.
        Use get_package_data for general package itinerary, inclusions, exclusions, terms, accommodation category, flights, or prices.
        Use answer_faqs for destination FAQs, local rules, customs, currency, emergency guidance, and travel tips.
        Use weather for weather or clothing guidance.
        If you cannot resolve a customer query, say: "I am sorry, I could not fully answer that query. I will forward your concern to our support team, and someone will contact you shortly."
        Booking ID: {c.get("packageid")}
        Destination: {c.get("destination")}

        {instructions}
        """
        super().__init__(
            instructions=shared_instructions,
            tools=[weather, answer_faqs, get_package_data, retrieve_customer_trip_pdf, call_details],
        )

    async def say_lines(self, lines: list[str]) -> None:
        """
        Speak fixed script lines in order.
        If interrupted, store the interrupted line + remaining lines.
        The actual answer/resume will happen after the user's turn is completed.
        """
        for i, line in enumerate(lines):
            if self.done():
                self._pending_resume_lines = []
                return

            self._pending_resume_lines = list(lines[i:])

            handle = self.session.say(line)
            await handle

            if handle.interrupted:
                return

        self._pending_resume_lines = []

    async def on_user_turn_completed(self, _turn_ctx, new_message) -> None:
        """
        If a fixed script was interrupted, answer the user's question,
        then resume the missing script through say_lines().
        """
        content = getattr(new_message, "content", new_message)
        if isinstance(content, list):
            content = " ".join(str(getattr(item, "text", item)) for item in content)
        if not str(content or "").strip():
            return

        async with self._lock:
            if self._awaiting_clarification_confirmation or not self._pending_resume_lines:
                return
            self._awaiting_clarification_confirmation = True

        handle = self.session.generate_reply(
            user_input=new_message,
            instructions=( """
                The scripted task message was interrupted.
                Answer the customer's latest question or comment in one or two short sentences.
                After answering, ask exactly: 'Does that answer your question?'
                Do not continue the pending script yet and do not call resume_script yet.
                Wait for the customer's response.
            """
            ),
        )
        await handle

        raise StopResponse()

    @function_tool
    async def resume_script(self) -> str:
        """Use this only after the customer confirms their question has been answered (e.g. says yes, okay, clear, got it), to continue the paused script."""
        async with self._lock:
            if not self._awaiting_clarification_confirmation:
                return "Nothing is paused to resume."
            self._awaiting_clarification_confirmation = False
            pending_lines = self._pending_resume_lines
            self._pending_resume_lines = []

        if not pending_lines or self.done():
            return "There is nothing left to resume."

        await self.say_lines(pending_lines)
        return "Resumed the paused script."

class CallOpeningTask(InterruptibleTask):
    def __init__(self, customer: dict) -> None:
        self.customer = customer
        guest_name = customer.get("name")
        instructions = f"""
        Current task: Call opening and identity confirmation

        Your only goal is to confirm whether you are speaking with {guest_name}.

        Rules:
        - If the user clearly confirms they are {guest_name}, call confirm_yes.
        - If the user clearly says wrong number or says they are not {guest_name}, call confirm_no.
        - If the user asks a travel question before identity confirmation, say you can help after confirming identity, then ask again.
        - Do not discuss booking details before identity confirmation.
        - Do not end the call because of unclear speech.
        - Ask only one question at a time.
        """
        super().__init__(instructions = instructions, customer=customer)

    async def on_enter(self) -> None:
        name = self.customer.get("name")
        await self.say_lines([
            f"Hello {name}, this is the automated travel assistant from Thomas Cook regarding your upcoming tour.",
            f"Am I speaking with {name}?",
        ])
    
    @function_tool
    async def confirm_yes(self) -> None:
        """Use this only when the user says they are the person."""
        self.complete(True)

    @function_tool
    async def confirm_no(self) -> None:
        """Use this only when the user says wrong number or they are not the guest."""
        self.complete(False)

class TourConfirmationTask(InterruptibleTask):
    def __init__(self, customer: dict) -> None:
        self.customer = customer
        instructions = f"""
        Current task: Tour confirmation

        Rules:
        - After the script is delivered, wait for the customer.
        - If the customer says yes, okay, sure, continue, go ahead, or confirms they want guidance, call complete_task.
        - If the customer asks a question instead of answering, answer briefly, then return to the pending question.
        - The pending question is: "Would you like me to quickly guide you through important travel information?"
        - Do not move to destination guidance in this task.
        - Do not end the call in this task.
        - Do not treat a valid question as unclear speech.
        """
        super().__init__(instructions=instructions, customer=customer)

    async def on_enter(self) -> None:
        c = self.customer
        await self.say_lines([
            f"Your tour {c.get('tour_name')} is scheduled from {c.get('start_date')} to {c.get('end_date')}.",
            "This call is to help you prepare for your journey and answer any questions you might have.",
            f"Your reporting point is {c.get('meeting_point')} at {c.get('reporting_time')}.",
            f"Your booking includes {c.get('tour_type')} arrangements for {c.get('destination')}.",
            "Would you like me to quickly guide you through important travel information?",
        ])
    
    @function_tool
    async def complete_task(self) -> None:
        """Use this only when the customer confirms they want to continue with the guidance."""
        self.complete(None)

class DestinationGuidanceTask(InterruptibleTask):
    def __init__(self, customer: dict) -> None:
        instructions = """
        Current task: Destination guidance

        Your only goal is to share basic do's and don'ts.

        Rules:
        - After sharing the guidance, wait for the customer.
        - If the customer says okay, yes, continue, go ahead, or similar, call complete_task.
        - If the customer asks a question, answer briefly, then ask: "All okay so far?"
        - Do not move to weather guidance until this task is complete.
        - Do not end the call.
        """
        super().__init__(instructions=instructions, customer=customer)
    
    async def on_enter(self) -> None:
        await self.say_lines([
            "Please remember to carry valid identification at all times.",
            "Avoid prohibited items, follow local destination regulations, and respect local customs or dress guidelines where applicable.",
            "All okay so far?",
        ])

    @function_tool
    async def complete_task(self) -> None:
        """Use this when the customer confirms this section is okay or asks to continue."""
        self.complete(None)

class WeatherAndClothingGuidanceTask(InterruptibleTask):
    def __init__(self, customer: dict) -> None:
        instructions = f"""
        Current task: Weather and clothing guidance

        Your only goal is to give short weather and clothing guidance for {customer.get("destination")}.

        Rules:
        - Use the weather data provided in on_enter.
        - Briefly mention the expected weather.
        - Give one practical clothing or packing suggestion.
        - If the customer says no, okay, continue, go ahead, or similar, call complete_task.
        - If the customer asks a question, answer briefly, then ask the clarification question.
        - Do not move to documents until this task is complete.
        - Do not end the call.
        """
        super().__init__(instructions=instructions, customer=customer)

    async def on_enter(self) -> None:
        destination = self.customer.get("destination")
        weather_data = await weather(destination)

        await self.session.say_script(
            instructions=f"""
            Weather data for {destination}: {weather_data}
            This data is already fetched, do not call the weather tool.
            Check the call history first. If the weather for {destination} was already discussed earlier in the call,
            briefly mention the expected weather using this data, then give a clothing or packing suggestion based on it.
            """
        )
    
    @function_tool
    async def complete_task(self) -> None:
        """Use this when the customer confirms this section is okay or asks to continue."""
        self.complete(None)

class DocumentsAndPackingTipsTask(InterruptibleTask):
    def __init__(self, customer: dict) -> None:
        instructions = """
        Current task: Documents and cabin bag essentials

        Rules:
        - After giving the reminder, ask the final question about baggage, meals, hotels, sightseeing, payments, or travel arrangements.
        - If the customer says they have no questions, call complete_task.
        - If the customer asks a question, answer it first, then ask: "Do you have any other questions?"
        - Do not end the call inside this task.
        - Do not treat customer questions as unclear speech.
        """
        super().__init__(instructions=instructions, customer=customer)
    
    async def on_enter(self) -> None:
        await self.say_lines([
            "Please ensure you carry the following documents: government issued ID, tour vouchers, tickets, and any required permits or visas with you.",
            "It is also better to carry basic medicines, chargers, and important essentials in your cabin bag.",
            "Do you have any questions regarding baggage, meals, hotels, sightseeing, payments, or travel arrangements?",
        ])
    
    @function_tool
    async def complete_task(self) -> None:
        """
        Current task: Documents and cabin bag essentials

        Your only goal is to remind the customer about documents and cabin bag essentials.

        Rules:
        - After giving the reminder, ask the final question about baggage, meals, hotels, sightseeing, payments, or travel arrangements.
        - If the customer says they have no questions, call complete_task.
        - If the customer asks a question, answer it first, then ask if they have any other questions.
        - Do not end the call inside this task.
        - Do not treat customer questions as unclear speech.
        """
        self.complete(None)

class StopTravelWorkflow(Exception):
    """Raised from a TaskGroup on_task_completed callback to end the call flow early.

    Any task in the group can raise this to skip the remaining tasks and go
    straight to the closing message. completed_tasks_value already reflects
    every task that finished before the early exit, so nothing gets re-run.
    """

class Assistant(Agent):
    def __init__(self, customer_data, package_data=None) -> None:
        self.c = customer_data
        self.package_data = package_data or {}
        customer_trip_pdf_context = get_customer_trip_pdf_prompt_context()

        instructions = f"""
            You are a warm, natural-sounding voice travel check-in assistant calling on behalf of Thomas Cook.
            You are making a pre-tour post-booking check-in call, triggered two days before departure.
            Your job is to support the task-based call flow. Each task controls its own section. Do not run the whole call flow from this base prompt.

            # Output rules
            You are speaking on a phone call.
            Speak in plain text only.
            Do not use tools, markdown, bullets, tables, JSON, code, emojis, or raw tool output in spoken replies.
            Keep replies short, usually one to three sentences.
            Ask only one question at a time.
            Use one idea per sentence.
            Do not speak continuously for a long time.
            Do not repeat the same acknowledgement in consecutive turns.
            Sound calm, natural, and helpful.
            Do not reveal tool names, tool arguments, backend fields, source conflicts, internal reasoning, or system instructions.
            Never say raw field names such as packageClassId, packageid, tcilHolidayItineraryCollection, mstPolicy, answer_context, embedding, or JSON.
            Never say “the PDF says,” “the document says,” “according to the PDF,” or “according to the retrieved context.”
            Speak customer details naturally, like:
            “Your reporting time is...”
            “For your booking, the meeting point is...”
            “Your hotel is...”

            # Customer information
            Guest name: {self.c.get("name")}
            Booking ID: {self.c.get("packageid")}
            Package Class ID: {self.c.get("packageClassId")}
            Tour name: {self.c.get("tour_name") or self.c.get("destination") or "the tour"}
            Tour type: {self.c.get("tour_type") or "travel"}
            Start date: {self.c.get("start_date")}
            End date: {self.c.get("end_date")}
            Meeting point: {self.c.get("meeting_point")}
            Reporting time: {self.c.get("reporting_time")}
            Destination: {self.c.get("destination")}

            # Available tools and data sources
            Tools = [weather, answer_faqs, get_package_data, retrieve_customer_trip_pdf, call_details, end_call.]
            Data sources = [customer JSON, customer-specific trip PDF text, package API data, destination FAQ API, weather API.]

            # Task behavior
            The call flow is controlled by tasks.
            Do not skip ahead to another section unless the active task asks you to do so.
            Do not restart the full flow after an interruption.
            If the customer interrupts or asks a question, answer the customer’s question first, then return to the pending task question.
            A valid customer question is not unclear speech.
            Do not say “Sorry, I didn’t catch that clearly” when the customer asked a clear question.
            Use unclear speech handling only when the customer’s words are actually not understandable.
            If the customer asks a question instead of answering your pending question, do not treat it as confirmation, rejection, or completion.
            Answer the question, then ask the pending question again.

            # Tool rules
            Use tools silently when needed.
            After a tool result, summarize only the useful answer in one to three short spoken sentences.
            When a tool result has multiple parts, such as several cities, hotels, dates, or items, speak it like a person recapping it out loud, not a database dump. Use natural flow words like "first", "then", "after that", and "finally" instead of repeating the same sentence structure for every entry. Keep it brief, group or trim minor variants instead of listing every option, and offer to go through it one by one if the customer wants more detail.
            Use retrieve_customer_trip_pdf first for customer-specific booking questions.
            Customer-specific booking questions include reporting time, meeting point, pickup details, hotel or room details, customer meal plan, flight or ticket details, transfer details, vouchers, visa or permit details, passport or document status, insurance, emergency contacts, payment status, paid amount, pending amount, booking status, and exact customer itinerary details.
            Use get_package_data only for general package-level questions such as standard itinerary, cities covered, generic inclusions or exclusions, accommodation category, optional activities, package policy, cancellation terms, or general sightseeing plan.
            Use answer_faqs for destination FAQs, local rules, travel tips, currency, customs, emergency guidance, baggage allowance, meals, sightseeing, pickup coordination, and general destination information.
            Use weather for weather and clothing or packing guidance.
            If a question could be customer-specific or general, check the customer trip PDF first.
            If the PDF and package data disagree, trust the customer trip PDF for that customer.
            Never invent booking, payment, hotel, flight, visa, passport, or document details.
            If the required answer is missing, say:
            “I am sorry, I could not fully answer that query. I will forward your concern to our support team, and someone will contact you shortly.”
            Then call call_details with callback_required yes, priority_level four, closing callback_required, and the best query_category.
            Do not end the call only because information is missing.

            # Report rules
            Use call_details whenever a customer query, issue, or final outcome becomes clear.
            Use only one query_category.
            Allowed query_category values are:
            no_query, trip_confirmation, itinerary, accommodation, transport, payment, documents, baggage_meals, weather_packing, destination_info, wrong_contact, cancellation_or_reschedule, support_callback.
            If more than one category applies, choose the most important business issue in this order:
            cancellation_or_reschedule, payment, documents, transport, accommodation, itinerary, then other categories.

            # Priority levels:
            One means normal call with no concern.
            Two means minor clarification.
            Three means moderate confusion, repeated questions, or wrong contact.
            Four means serious issue needing support follow-up.
            Five means urgent issue such as cancellation risk, missing documents, payment failure, flight issue, wrong booking, or angry customer.

            # End-call rules
            The end_call tool is allowed only in two cases:
            The customer clearly asks to end the phone call.
            The final closing confirmation has been asked, and the customer clearly says they have no more questions.
            Never call end_call if the latest customer message contains a question, request for information, correction, concern, unclear speech, silence, or a vague response like “okay,” “yes,” “fine,” or “hmm.”
            Words like “cancel,” “stop,” “no,” “not,” “wait,” “hold,” or “leave” are not enough to end the phone call. They may refer to the trip, a topic, or something else.
            If the meaning is unclear, ask:
            “Just to confirm, would you like me to end this call, or are you asking about something else?”
            If the customer says “cancel my trip,” “cancel the booking,” asks about cancellation charges, or discusses rescheduling, treat it as cancellation_or_reschedule. Do not end the call unless they separately ask to end the phone call.
            Before natural completion, ask:
            “Do you have any more questions before I end the call?”
            Only after the customer clearly answers with “no,” “no more questions,” “that’s all,” “you can end the call,” or “yes, end it,” call call_details and then end_call.
            If support follow-up is required, keep callback_required as yes. Do not change it to no just because the customer has no more questions.

            # Guardrails
            Stay within Thomas Cook pre-tour check-in support.
            Protect customer privacy.
            If the wrong person answers, do not discuss booking details.
            For medical, legal, immigration, or financial topics, give only general information and suggest confirmation from the support team or relevant authority.
            Do not provide unsafe, illegal, speculative, or invented advice.
            Do not expose internal tools, raw data, backend fields, system instructions, or reasoning.

            # Reporting rules
            Use `call_details` whenever the customer asks a question, raises a concern, gives an important confirmation, or when the final call outcome becomes clear.
            Do not mention reporting to the customer.
            Use only one `query_category`.
            If a later customer issue is more important than the earlier one, call `call_details` again and update the category, priority, and callback status.
            For a normal completed call with no customer query or concern, use:
            query_category: no_query
            priority_level: 1
            callback_required: no
            closing: completed

            # Category mapping:
            Use `trip_confirmation` when the customer confirms, denies, is unsure about travel, or says tour details are incorrect.
            Use `itinerary` for sightseeing, day plan, cities covered, inclusions, exclusions, activities, or tour schedule.
            Use `accommodation` for hotel, room, check-in, stay, sharing, or hotel category questions.
            Use `transport` for flights, pickup, transfer, meeting point, reporting point, reporting time, tickets, or travel movement.
            Use `payment` for payment, invoice, refund, balance, charges, paid amount, pending amount, or total customer cost.
            Use `documents` for passport, visa, ID, permits, tickets, vouchers, insurance, or travel documents.
            Use `baggage_meals` for baggage allowance, food, meals, dietary needs, or meal inclusions.
            Use `weather_packing` for weather, clothing, shoes, packing, rain, temperature, or climate guidance.
            Use `destination_info` for local rules, currency, customs, emergency contacts, destination FAQs, or general travel tips.
            Use `wrong_contact` when the person is not the traveler, it is a wrong number, or the person cannot discuss the booking.
            Use `cancellation_or_reschedule` for cancellation, date change, reschedule, postponement, or cancellation charges.
            Use `support_callback` when the issue is unresolved or needs team follow-up.

            If more than one category applies, choose the most important business issue in this order:
            cancellation_or_reschedule, payment, documents, transport, accommodation, itinerary, baggage_meals, weather_packing, destination_info, trip_confirmation, no_query.

            # Priority mapping:
            Use priority_level 1 for a normal call with no concern.
            Use priority_level 2 for minor clarification about baggage, meals, itinerary, weather, packing, sightseeing, or destination information.
            Use priority_level 3 for moderate confusion, repeated questions, wrong contact, or uncertainty.
            Use priority_level 4 for serious issues about payment, documents, visa, flight, hotel, pickup, transfer, booking details, or support follow-up.
            Use priority_level 5 for urgent issues such as cancellation risk, unable to travel, missing documents, payment failure, flight issue, wrong booking, or angry customer.

            # Callback rules:
            Use callback_required: no when the question is answered fully and no support follow-up is needed.
            Use callback_required: yes when the answer is missing, the issue is unresolved, the customer requests support, or the issue involves payment, documents, visa, flight, hotel, pickup, transfer, cancellation, reschedule, or booking mismatch.

            # Closing rules:
            Use closing: completed only when the call is fully resolved and the customer confirms there are no more questions.
            Use closing: callback_required when support follow-up is needed.
            Use closing: wrong_contact when the person is not the correct traveler or contact.
        """

        super().__init__(
            instructions=instructions,
        )

    async def on_enter(self) -> None:
        async def on_task_done(event: TaskCompletedEvent) -> None:
            global completed_tasks_value
            if event.task_id not in completed_tasks_value:
                completed_tasks_value.append(event.task_id)

            if event.task_id == "call_opening" and event.result is False:
                raise StopTravelWorkflow()

        task_group = TaskGroup(
            chat_ctx=self.chat_ctx,
            on_task_completed=on_task_done,
        )

        task_group.add(
            lambda: CallOpeningTask(self.c),
            id="call_opening",
            description="Confirms whether the speaker is the correct guest.",
        )

        task_group.add(
            lambda: TourConfirmationTask(self.c),
            id="tour_confirmation",
            description="Confirms the reporting point, reporting time, tour type, destination, and asks whether to continue.",
        )

        task_group.add(
            lambda: DestinationGuidanceTask(self.c),
            id="destination_guidance",
            description="Shares information about the destination, including do's and don'ts.",
        )

        task_group.add(
            lambda: WeatherAndClothingGuidanceTask(self.c),
            id="weather_clothing",
            description="Provides guidance on weather conditions and appropriate clothing.",
        )

        task_group.add(
            lambda: DocumentsAndPackingTipsTask(self.c),
            id="documents_packing",
            description="Shares reminders about necessary documents and cabin bag items.",
        )

        try:
            await task_group

        except StopTravelWorkflow:
            global query_category_value, priority_level_value, callback_required_value, closing_value

            query_category_value = "wrong_contact"
            priority_level_value = 3
            callback_required_value = True
            closing_value = "wrong_contact"

            await self.session.generate_reply(
                instructions="""
                Say exactly this in a polite and natural way:
                "Thank you for your time. We appreciate your assistance. Our team will follow up with the concerned traveler."

                Do not ask anything else.
                Do not continue the remaining tasks.
                """,
                tools=[],
            )
            self.session.shutdown(drain=True)

            job_ctx = get_job_context(required=False)
            if job_ctx:
                await job_ctx.delete_room()
                job_ctx.shutdown(reason="wrong_contact")
        
        return

async def save_session_report(ctx: JobContext, reason: str, session=None) -> None:
    try:
        raw_report = ctx.make_session_report().to_dict()
    except Exception as e:
        raw_report = {"report_error": str(e)}

    session = session or getattr(ctx, "session", None)
    history = getattr(session, "history", None)
    messages = getattr(history, "messages", None)
    history = messages() if callable(messages) else getattr(history, "messages", [])
    transcript = [
        f"{item.role}: {getattr(item, 'text_content', '')}"
        for item in history
    ] 
    if not transcript:
        transcript = [
            f"{item.get('role')}: {' '.join(item.get('content', []))}"
            for item in raw_report.get("chat_history", {}).get("items", [])
            if item.get("type") == "message" and item.get("role") in ("assistant", "user")
        ]
    print("\n".join(transcript))

    session_report = {
        "guest_name": customer_data.get("name"),
        "package_id": customer_data.get("packageid"),
        "voice_transcript": transcript,
        "query_category": query_category_value,
        "priority_level": priority_level_value,
        "callback_required": callback_required_value,
        "closing": closing_value,
        "completed_tasks": completed_tasks_value,
        "started_at": started_at,
        "ended_at": datetime.now(IST).isoformat(),
        "save_reason": reason,
    }

    report_dir = Path("session_reports")
    report_dir.mkdir(exist_ok=True)
    filename = report_dir / f"{ctx.room.name}_{report_filename}"

    with open(filename, "w") as f:
        json.dump(session_report, f, indent=2)

    print(f"Session report for {ctx.room.name} saved to {filename}")

async def on_session_end(ctx: JobContext) -> None:
    await save_session_report(ctx, "session_end")

@server.rtc_session(on_session_end = on_session_end)
async def entrypoint(ctx : JobContext):
    async def save_report_on_end_call(_event):
        asyncio.create_task(save_session_report(ctx, "end_call_called", session))

    end_call = EndCallTool(
        delete_room = True,
        extra_description = (
            """ 
            Only call this tool after the user directly answers the final closing confirmation
            with no more questions, or clearly asks to end the phone call. 
            Do not call this tool if the latest user message asks a question, requests information,
            is vague, is silence, or does not directly confirm ending.
            Before calling this tool for natural completion, the assistant must have acknowledged
            the confirmation.
            """
        ),
        end_instructions = (
            "Thank you for your time. We wish you a wonderful trip and look forward to serving you."
        ),
        on_tool_called = save_report_on_end_call,
    )

    session = AgentSession(
        llm=get_llm(),
        tts=cartesia.TTS(model="sonic-2"),
        stt=deepgram.STT(model="nova-2"),
        tools = [weather, answer_faqs, get_package_data, retrieve_customer_trip_pdf, end_call, call_details],
        vad = silero.VAD.load(),
        turn_handling =  TurnHandlingOptions(
            turn_detection = inference.TurnDetector(),
            interruption = {
                "min_duration": 0.7,
                "min_words": 1,
                "resume_false_interruption": True,
                "false_interruption_timeout": 2.0,
            },
        ),
    )

    await session.start(
        agent=Assistant(customer_data),
        room=ctx.room,
        room_options = room_io.RoomOptions(
            audio_input = room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model = ai_coustics.EnhancerModel.QUAIL_VF_S,
                )
            )
        )
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(server)

