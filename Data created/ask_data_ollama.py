#!/usr/bin/env python3
"""
FastColis — US 5.1 : interrogation NL légère (sans AgentExecutor).

Pipeline en 2 appels Ollama, sans langchain_experimental / Transformers / PyTorch :
  1. Le LLM génère UNE expression Pandas sur `df`.
  2. On l'exécute localement (eval) sur le DataFrame chargé.
  3. Le LLM reformule le résultat chiffré en une phrase française.

Prérequis :
    ollama serve && ollama pull qwen2.5
    pip3 install pandas langchain-ollama langchain-core
    python3 generate_test_csv.py

Usage :
    python3 ask_data_ollama.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "donnees_finals.xlsx"
DATA_PATH_REC = Path(__file__).resolve().parent / "reclamations_laposte_2025-2026 (1).csv"
OLLAMA_MODEL = "qwen2.5"

CODE_SYSTEM = """Tu es un traducteur question métier → Pandas.
Tu as accès à deux DataFrames : `df` et `df_rec`. Tu ne fais AUCUN calcul toi-même.

1. `df` (Satisfaction Colis) :
- id_colis, pays, code_produit (COLIS_EXPRESS, COLIS_STANDARD, COLIS_FRAGILE, COLIS_VOLUMINEUX, PAK_*), score_delai, score_livreur, score_etat_colis, score_service_client, score_global.

2. `df_rec` (Réclamations La Poste) :
- id_reclamation, produit (Colissimo, Chronopost...), categorie, sous_categorie, delai_traitement_jours, montant_indemnisation_eur, note_satisfaction.

Règles Pandas :
- Tu peux utiliser `df`, `df_rec`, ou combiner/croiser les deux.
- Attention : df_rec n'a pas de colonne id_colis ni de colonne pays (uniquement France).
- RÈGLE DES DICTIONNAIRES (CRITIQUE) : Si tu renvoies un dictionnaire, les clés DOIVENT IMPÉRATIVEMENT être des chaînes de caractères littérales entre guillemets (ex: {'repartition_pays': df['pays'].value_counts(), 'total_reclamations': len(df_rec)}). Ne JAMAIS utiliser df['colonne'] ou un objet Pandas comme clé de dictionnaire.
- Pour les calculs de taux ou de corrélation globale : génère un dictionnaire avec des clés 'str'.
- N'essaie JAMAIS de faire pd.merge(df, df_rec) sur 'id_colis' ou 'pays'.
- Réponds UNIQUEMENT par une expression Python Pandas commençant par df, df_rec ou {.
- Pas de markdown, pas de commentaires, pas d'import, pas de print, pas d'explication.
- Ne JAMAIS utiliser d'assignation comme `res =` ou `resultat =`. Commence directement par `df`, `df_rec` ou `{`.
"""

ANSWER_SYSTEM = """Tu es un analyste FastColis.
À partir du ou des résultats Pandas déjà calculés, rédige une réponse en français claire, exhaustive et bien structurée.
Si les données ne permettent pas une jointure parfaite (ex. absence d'id_colis commun), synthétise les données disponibles dans chaque table séparément et explique-le poliment dans ta réponse.
Exploite l'ensemble des clés ou résultats fournis. N'invente rien. N'ajoute pas de code."""

FORBIDDEN_TOKENS = (
    "import ",
    "__",
    "open(",
    "exec(",
    "eval(",
    "os.",
    "sys.",
    "subprocess",
    "pathlib",
    "globals",
    "locals",
    "compile(",
    "input(",
    "breakpoint",
)


def load_dataframe(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        fallback = path.with_suffix(".csv")
        if fallback.exists():
            path = fallback
        else:
            print(
                f"Fichier introuvable : {path}\n",
                file=sys.stderr,
            )
            sys.exit(1)
    
    if path.suffix == ".xlsx":
        df = pd.read_excel(path, **kwargs)
    else:
        df = pd.read_csv(path, **kwargs)
        
    print(f"[ask_data_ollama] Données chargées : {path.name} ({len(df)} lignes)")
    print(df.head().to_string(index=False))
    print("-" * 72)
    return df


def build_llm():
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        print(
            "Installez : pip3 install langchain-ollama langchain-core pandas\n"
            f"Détail : {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ChatOllama seul : pas d'agent, pas de transformers.
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        num_predict=512,
    )


def extraire_expression(raw: str) -> str:
    """Retire fences markdown et extrait l'expression ou le dictionnaire."""
    text = raw.strip()
    text = re.sub(r"```(?:python)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        line_s = line.strip()
        line_s = re.sub(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*", "", line_s)
        if line_s.startswith("df") or line_s.startswith("df_rec") or line_s.startswith("{"):
            # Recomposer tout le bloc au cas où c'est un dict sur plusieurs lignes
            lines[i] = line_s
            return "\n".join(lines[i:]).strip()
    return text.strip() if text else ""


def expression_autorisee(expr: str) -> None:
    """Refuse tout ce qui n'est pas une expression Pandas ou un dict d'expressions."""
    if "df" not in expr and "df_rec" not in expr and "{" not in expr:
        raise ValueError(f"Expression refusée (doit contenir df, df_rec ou {{) : {expr}")
    lowered = expr.lower()
    for token in FORBIDDEN_TOKENS:
        if token in lowered:
            raise ValueError(f"Expression refusée (jeton interdit) : {expr}")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Syntaxe Pandas invalide : {expr}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError(f"Import interdit : {expr}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"Attribut interdit : {expr}")


def formater_resultat(value) -> str:
    if isinstance(value, dict):
        return "\n".join(f"- {k} : {formater_resultat(v)}" for k, v in value.items())
    if isinstance(value, pd.Series):
        return value.round(2).to_string() if value.dtype.kind in "fc" else value.to_string()
    if isinstance(value, pd.DataFrame):
        return value.round(2).to_string(index=False)
    if isinstance(value, float):
        return str(round(value, 2))
    return str(value)


def poser_question(llm, df: pd.DataFrame, df_rec: pd.DataFrame, question: str, max_retries: int = 3) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    print(f"\n>>> Question : {question}")

    messages = [
        SystemMessage(content=CODE_SYSTEM),
        HumanMessage(content=question),
    ]

    valeur = None
    expr = ""

    for attempt in range(1, max_retries + 1):
        code_msg = llm.invoke(messages)
        expr = extraire_expression(getattr(code_msg, "content", str(code_msg)))

        try:
            expression_autorisee(expr)
            valeur = eval(expr, {"__builtins__": {}}, {"df": df, "df_rec": df_rec, "pd": pd, "np": np})  # noqa: S307
            break  # Succès !
        except Exception as exc:
            print(f"[Essai {attempt}/{max_retries}] Erreur Pandas : {type(exc).__name__}: {exc}")
            if attempt == max_retries:
                raise exc
            
            error_msg = (
                f"L'expression Pandas suivante a échoué avec l'erreur {type(exc).__name__}: {exc}\n"
                f"Code erroné : {expr}\n"
                "Rappel : Les clés de dictionnaire doivent être des chaînes littérales entre guillemets (ex: 'cle': val). "
                "Ne mets jamais df['colonne'] en clé. Réponds UNIQUEMENT avec l'expression corrigée."
            )
            messages.append(code_msg)
            messages.append(HumanMessage(content=error_msg))

    resultat = formater_resultat(valeur)

    phrase_msg = llm.invoke(
        [
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(
                content=(
                    f"Question : {question}\n"
                    f"Code Pandas : {expr}\n"
                    f"Résultat calculé : {resultat}"
                )
            ),
        ]
    )
    reponse = getattr(phrase_msg, "content", str(phrase_msg)).strip()
    print(f"\n<<< Réponse : {reponse}\n")
    return reponse


def boucle_interactive(llm, df: pd.DataFrame, df_rec: pd.DataFrame) -> None:
    print("=== FastColis — interrogation rapide (Ollama / qwen2.5) ===")
    print("Exemples : Satisfaction SAV en France ? | Score global par pays ?")
    print("Tapez quit, exit ou q pour quitter.\n")

    while True:
        try:
            question = input("Question FastColis > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFin de session.")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Fin de session.")
            break
        try:
            poser_question(llm, df, df_rec, question)
        except Exception as exc:
            print(f"Erreur : {exc}", file=sys.stderr)


def main() -> None:
    df = load_dataframe(DATA_PATH)
    df_rec = load_dataframe(DATA_PATH_REC, sep=';')
    print(f"[ask_data_ollama] Connexion Ollama, modèle={OLLAMA_MODEL} …")
    llm = build_llm()
    boucle_interactive(llm, df, df_rec)


if __name__ == "__main__":
    main()
