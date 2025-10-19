"""
CrewAI Agents Framework 

This file provides a framework for the agents outlined in the flowchart. It is very high level
right now.

"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Protocol, Tuple
import os
import json
import datetime

# -- Data models -------------------------------------------------------------

@dataclass
class ProfessorProfile:
    professor_id: str
    name: Optional[str] = None
    department: Optional[str] = None
    courses: List[str] = field(default_factory=list)
    topics_by_week: Dict[str, List[str]] = field(default_factory=dict)  # ISO week -> topics
    preferences: Dict[str, Any] = field(default_factory=dict)  # e.g. demo complexity, safety notes


@dataclass
class DemoCandidate:
    demo_id: Optional[str]
    title: str
    description: str
    source: str  # e.g. 'internal_document', 'youtube', 'generated'
    required_items: List[str] = field(default_factory=list)
    estimated_prep_time_mins: Optional[int] = None
    confidence_score: float = 0.0


@dataclass
class InventoryItem:
    sku: str
    name: str
    quantity: int
    location: Optional[str] = None


# -- Protocols / interfaces for platform integrations ------------------------
class SyllabusPortal(Protocol):
    def list_uploaded_syllabuses(self, professor_id: str) -> List[str]:
        ...

    def download_syllabus(self, syllabus_id: str) -> bytes:
        ...


class DemoDocumentStore(Protocol):
    def search(self, query: str) -> List[DemoCandidate]:
        ...


class WooCommerceClient(Protocol):
    def list_recent_orders(self, days: int = 30) -> List[Dict[str, Any]]:
        ...

    def create_order(self, order_payload: Dict[str, Any]) -> Dict[str, Any]:
        ...


class InventoryDB(Protocol):
    def lookup_items(self, item_names: List[str]) -> List[InventoryItem]:
        ...


class EmailSender(Protocol):
    def send_email(self, to: str, subject: str, body: str, attachments: Optional[List[Any]] = None) -> None:
        ...


class Retriever(Protocol):
    """Retrieval interface for RAG / vector store.
    """
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        ...


class ChatBotInterface(Protocol):
    def chat_with_professor(self, professor_id: str, initial_prompt: str) -> Dict[str, Any]:
        ...


# -- Base Agent --------------------------------------------------------------
class Agent:
    def __init__(self, name: str):
        self.name = name

    def run(self, *args, **kwargs):
        raise NotImplementedError("Agent.run must be implemented by subclasses")


# -- Agents ------------------------------------------------------------------
class SyllabusProfileAgent(Agent):
    """Generates a ProfessorProfile from uploaded syllabuses.

    Responsibilities:
    - download syllabuses for a professor
    - parse syllabus content (structure, weekly topics, assignments)
    - produce a normalized ProfessorProfile
    """
    def __init__(self, portal: SyllabusPortal):
        super().__init__("syllabus_profile_agent")
        self.portal = portal

    def parse_syllabus_text(self, raw_bytes: bytes) -> Dict[str, Any]:
        # TODO: replace with actual PDF/Word parsing, NLP to extract weeks/topics
        text = raw_bytes.decode(errors='ignore')[:10000]
        return {"text_sample": text}

    def run(self, professor_id: str) -> ProfessorProfile:
        syllabus_ids = self.portal.list_uploaded_syllabuses(professor_id)
        profile = ProfessorProfile(professor_id=professor_id)

        for s_id in syllabus_ids:
            raw = self.portal.download_syllabus(s_id)
            parsed = self.parse_syllabus_text(raw)
            # TODO: NLP that splits into ISO-week -> topics mapping
            week_key = datetime.date.today().isocalendar()  # placeholder
            profile.courses.append(s_id)
            profile.topics_by_week[str(week_key)] = [parsed.get('text_sample', '')[:200]]

        # Extract/guess name, dept etc. from parsed content if available
        return profile


class DemoFinderAgent(Agent):
    """Finds relevant demos based on the current teaching week and document store + WooCommerce trends.

    Responsibilities:
    - determine what is being taught this week from ProfessorProfile
    - search internal demo document using keywords
    - optionally consult WooCommerce for purchase trends
    """
    def __init__(self, demo_store: DemoDocumentStore, woocommerce: Optional[WooCommerceClient] = None):
        super().__init__("demo_finder_agent")
        self.demo_store = demo_store
        self.woocommerce = woocommerce

    def score_candidate(self, candidate: DemoCandidate, topics: List[str]) -> float:
        # Simple heuristic scoring -- replace with better model if desired
        score = 0.0
        for t in topics:
            if t.lower() in candidate.title.lower() or t.lower() in candidate.description.lower():
                score += 0.6
        # bump score if candidate recently ordered on WooCommerce (placeholder)
        return min(score, 1.0)

    def run(self, profile: ProfessorProfile, iso_week: str) -> List[DemoCandidate]:
        topics = profile.topics_by_week.get(iso_week, [])
        if not topics:
            return []

        # search demo documents for each topic
        candidates: List[DemoCandidate] = []
        for topic in topics:
            res = self.demo_store.search(topic)
            for r in res:
                r.confidence_score = self.score_candidate(r, topics)
                candidates.append(r)

        # Optionally augment with WooCommerce ordering patternserns
        if self.woocommerce is not None:
            orders = self.woocommerce.list_recent_orders(days=90)
            # TODO: analyze orders to influence confidence/priority

        # sort by confidence
        candidates.sort(key=lambda c: c.confidence_score, reverse=True)
        return candidates


class ProfessorChatAgent(Agent):
    """If DemoFinderAgent finds no good matches, open a chat with the professor
    to narrow requirements and re-run the search. If still no matches, expand to
    web searches and generate a demo that matches inventory.
    """
    def __init__(self,
                 chat_interface: ChatBotInterface,
                 demo_store: DemoDocumentStore,
                 retriever: Optional[Retriever],
                 inventory_db: InventoryDB):
        super().__init__("professor_chat_agent")
        self.chat_interface = chat_interface
        self.demo_store = demo_store
        self.retriever = retriever
        self.inventory_db = inventory_db

    def run(self, profile: ProfessorProfile, iso_week: str, prior_candidates: List[DemoCandidate]) -> Tuple[List[DemoCandidate], Optional[str]]:
        # If we already have strong candidates, simply return them
        if prior_candidates and prior_candidates[0].confidence_score >= 0.7:
            return prior_candidates, None

        # otherwise, ask professor for clarification via chat interface
        prompt = "Hi — could you share more details about the kind of demo you'd like for the week's topics?"
        chat_result = self.chat_interface.chat_with_professor(profile.professor_id, prompt)
        # Expect chat_result to include keywords or free text
        user_keywords = chat_result.get("keywords", []) or [chat_result.get("transcript", "")]

        # Re-search the document store with refined keywords
        refined_candidates: List[DemoCandidate] = []
        for k in user_keywords:
            refined_candidates.extend(self.demo_store.search(k))

        # If still empty, use RAG retriever (e.g. YouTube transcripts, external resources)
        if not refined_candidates and self.retriever is not None:
            retrievals = self.retriever.retrieve(' '.join(user_keywords), top_k=10)
            # retrievals -> list of (text_snippet, score). Convert to DemoCandidate placeholders
            for text, score in retrievals:
                cand = DemoCandidate(demo_id=None, title=(text[:60] + '...'), description=text, source='external_rag', confidence_score=score)
                refined_candidates.append(cand)

        # For each candidate, cross-reference inventory to ensure it can be built at workplace
        validated: List[DemoCandidate] = []
        for c in refined_candidates:
            # naive item extraction: assume required_items already set, or do NLP extraction here
            items_needed = c.required_items
            inv = self.inventory_db.lookup_items(items_needed) if items_needed else []
            if items_needed and len(inv) < len(items_needed):
                # can't fully build from inventory; keep but lower confidence
                c.confidence_score *= 0.6
            validated.append(c)

        # If no validated candidates, optionally generate a novel demo (stub)
        if not validated:
            gen = DemoCandidate(demo_id=None, title='Generated demo proposal', description='Auto-generated demo based on professor input', source='generated', required_items=[], confidence_score=0.4)
            # cross-ref inventory for generated demo
            gen.required_items = []  # TODO: generate required items list
            validated.append(gen)

        # sort and return
        validated.sort(key=lambda c: c.confidence_score, reverse=True)
        return validated, chat_result.get('transcript')


class OrderAndNotificationAgent(Agent):
    """Sends orders to WooCommerce if a candidate corresponds to an existing demo that we
    can order, or sends an email request for building a custom demo.
    """
    def __init__(self, woocommerce: WooCommerceClient, email_sender: EmailSender, inventory_db: InventoryDB):
        super().__init__("order_notification_agent")
        self.woocommerce = woocommerce
        self.email_sender = email_sender
        self.inventory_db = inventory_db

    def run(self, professor: ProfessorProfile, chosen_demo: DemoCandidate, contact_email: str) -> Dict[str, Any]:
        # If demo is from internal site and has an orderable product id (demo_id), create a WooCommerce order
        if chosen_demo.source == 'internal_document' and chosen_demo.demo_id:
            payload = {
                # Fill with required WooCommerce order payload fields
                'line_items': [{'sku': chosen_demo.demo_id, 'quantity': 1}],
                'customer': {'email': contact_email},
                'note': f'Demo order for prof {professor.professor_id} - {chosen_demo.title}'
            }
            result = self.woocommerce.create_order(payload)
            return {'status': 'ordered', 'woocommerce_response': result}

        # Otherwise email a request to the demo team with details
        subject = f"Demo request: {chosen_demo.title} for Prof {professor.name or professor.professor_id}"
        body = f"Professor {professor.name or professor.professor_id} requested a demo:\n\nTitle: {chosen_demo.title}\nDescription: {chosen_demo.description}\nRequired items: {chosen_demo.required_items}\n"
        self.email_sender.send_email(to='demos-team@example.com', subject=subject, body=body)
        return {'status': 'emailed', 'email_to': 'demos-team@example.com'}


class HelpBotAgent(Agent):
    """A helper chatbot that looks up resources for a specific demo (YouTube/TAMU videos, other universities).
    It uses a Retriever/RAG for transcript-level search and returns curated links.
    """
    def __init__(self, retriever: Retriever):
        super().__init__("help_bot_agent")
        self.retriever = retriever

    def run(self, demo_query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        hits = self.retriever.retrieve(demo_query, top_k=top_k)
        # convert to structured results
        results = []
        for snippet, score in hits:
            results.append({'snippet': snippet, 'score': score})
        return results


# -- Orchestration example --------------------------------------------------
class AgentsOrchestrator:
    def __init__(self,
                 syllabus_agent: SyllabusProfileAgent,
                 demo_finder: DemoFinderAgent,
                 chat_agent: ProfessorChatAgent,
                 order_agent: OrderAndNotificationAgent,
                 help_bot: HelpBotAgent):
        self.syllabus_agent = syllabus_agent
        self.demo_finder = demo_finder
        self.chat_agent = chat_agent
        self.order_agent = order_agent
        self.help_bot = help_bot

    def process_professor_for_week(self, professor_id: str, iso_week: str, contact_email: str) -> Dict[str, Any]:
        profile = self.syllabus_agent.run(professor_id)
        candidates = self.demo_finder.run(profile, iso_week)

        if not candidates:
            candidates, transcript = self.chat_agent.run(profile, iso_week, candidates)
        else:
            transcript = None

        # pick top candidate (application logic may be more complex)
        if candidates:
            chosen = candidates[0]
            result = self.order_agent.run(profile, chosen, contact_email)
        else:
            chosen = None
            result = {'status': 'no-candidates-found'}

        return {
            'professor_id': professor_id,
            'chosen_demo': chosen,
            'result': result,
            'chat_transcript': transcript
        }


# -- Example usage (stubs) --------------------------------------------------
if __name__ == '__main__':
    # Here we would inject concrete implementations of the protocol interfaces.
    # For now, we provide simple in-memory stubs for local testing.

    class DummyPortal:
        def list_uploaded_syllabuses(self, professor_id):
            return ['syll1']
        def download_syllabus(self, syllabus_id):
            return b"Week 1: Intro to waves\nWeek 2: Interference\n".ljust(4000, b' ')

    class DummyDemoStore:
        def search(self, query: str):
            return [DemoCandidate(demo_id='demo-waves-1', title='Wave interference demo', description='Use ripple tank', source='internal_document', required_items=['ripple tank'], confidence_score=0.5)]

    class DummyWooCommerce:
        def list_recent_orders(self, days=30):
            return []
        def create_order(self, payload):
            return {'order_id': 'ORD123', 'payload': payload}

    class DummyInventory:
        def lookup_items(self, item_names):
            return [InventoryItem(sku='SKU1', name=n, quantity=5) for n in item_names]

    class DummyEmail:
        def send_email(self, to, subject, body, attachments=None):
            print(f"SENT EMAIL to {to}: {subject}\n{body[:200]}")

    class DummyRetriever:
        def retrieve(self, query, top_k=5):
            return [(f"YouTube transcript snippet about {query}", 0.8)]

    class DummyChat:
        def chat_with_professor(self, professor_id, initial_prompt):
            return {'transcript': 'Looking for a demo about wave interference using sound', 'keywords': ['wave interference', 'sound demo']}

    portal = DummyPortal()
    demo_store = DummyDemoStore()
    woocommerce = DummyWooCommerce()
    inventory = DummyInventory()
    emailer = DummyEmail()
    retriever = DummyRetriever()
    chat_interface = DummyChat()

    syllabus_agent = SyllabusProfileAgent(portal)
    demo_finder = DemoFinderAgent(demo_store, woocommerce)
    chat_agent = ProfessorChatAgent(chat_interface, demo_store, retriever, inventory)
    order_agent = OrderAndNotificationAgent(woocommerce, emailer, inventory)
    help_bot = HelpBotAgent(retriever)

    orch = AgentsOrchestrator(syllabus_agent, demo_finder, chat_agent, order_agent, help_bot)

    out = orch.process_professor_for_week(professor_id='prof123', iso_week=str(datetime.date.today().isocalendar()), contact_email='prof@example.com')
    print(json.dumps(out, default=str, indent=2))
