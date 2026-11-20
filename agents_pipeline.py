import os
import re
from google import genai
from google.genai import types  # type: ignore
from google.genai.errors import ServerError  # type: ignore


MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")


def get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Aucune clé GOOGLE_API_KEY ou GEMINI_API_KEY trouvée dans l'environnement.")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta"),
    )


def _clean_html(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:html)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _generate_text(client: genai.Client, system_instruction: str, prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            ),
        )
        return (response.text or "").strip()
    except ServerError as exc:
        raise RuntimeError(f"API Gemini temporairement indisponible (503) : {exc}") from exc


def run_three_agents(user_prompt: str) -> dict[str, str]:
    client = get_client()

    # 1. Agent Architecte
    plan = _generate_text(
        client,
        system_instruction=(
            "Tu es un Architecte Produit & UI. En 3 à 5 lignes maximum en français, "
            "décris la structure, les champs interactifs et la logique du mini-outil demandé. "
            "Sois concret et concis."
        ),
        prompt=f"Besoin utilisateur : {user_prompt}",
    )

    # 2. Agent Développeur
    raw_html = _generate_text(
        client,
        system_instruction=(
            "Tu es un Développeur Frontend Senior. Génère UN SEUL fichier HTML complet et autonome "
            "incluant Tailwind CSS via CDN (https://cdn.tailwindcss.com) et le JavaScript nécessaire "
            "dans une balise <script>. Le rendu doit être propre, responsive, en français, sans dépendance "
            "externe cassée. Réponds UNIQUEMENT avec le code HTML brut, sans markdown ni commentaire autour."
        ),
        prompt=(
            f"Spécification utilisateur : {user_prompt}\n"
            f"Plan validé par l'Architecte : {plan}"
        ),
    )
    html = _clean_html(raw_html)

    # 3. Agent Contrôleur Qualité
    qa_report = _generate_text(
        client,
        system_instruction=(
            "Tu es un Contrôleur Qualité Frontend. En 2 à 3 lignes maximum en français, "
            "indique ce qui a été vérifié (structure HTML, interactivité JS, responsive mobile) "
            "et signale si le code est prêt à l'emploi."
        ),
        prompt=f"Vérifie ce code généré pour '{user_prompt}' :\n\n{html[:4000]}",
    )

    return {
        "plan": plan,
        "html": html,
        "qaReport": qa_report,
    }