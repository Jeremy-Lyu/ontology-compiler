"""
Multi-agent orchestrator for BOM RAG skill.
It mirrors the logic from deepseek_tools.bom_analyzer.main but is callable from a Skill.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

from deepseek_tools.common import load_config_env, init_deepseek_client
from deepseek_tools.bom_analyzer.session import AnalysisSession
from deepseek_tools.bom_analyzer.agents import PlannerAgent, AnalystAgent, VerifierAgent
from deepseek_tools.bom_analyzer.relevance_gate import RelevanceGate
from ontology_compiler.services.ontology_context_loader import OntologyContextLoader

class MultiAgentOrchestrator:
    """Encapsulates the multi‑agent analysis loop.

    Parameters are supplied via a ``context`` dictionary. Only a subset of
    arguments used by the original CLI are required.
    """

    def __init__(self):
        pass

    def run(self, context: Dict[str, Any]):
        """Execute the analysis workflow.

        Expected keys in ``context``:
            - ``project_dir`` (str): root of the repository.
            - ``ontology_run_dir`` (str): path to ontology run output.
            - ``master_ontology_dir`` (str, optional): alternative source.
            - ``csv_files`` (List[str]): list of CSV/Excel/JSON files.
            - ``background`` (str): user‑provided background description.
            - ``question`` (str): analysis question.
            - ``max_rounds`` (int, default 4)
            - ``converge_threshold`` (float, default 0.82)
            - ``max_csv_rows`` (int, default 300)
            - ``output_dir`` (str, optional)
            - ``local_only`` (bool, default False)
            - ``temperature`` (float, default 0.15)
            - ``max_tokens`` (int, default 8000)
        Returns a dict with ``status``, ``final_findings`` and ``output_dir``.
        """
        from deepseek_tools.bom_analyzer.main import main as bom_analyzer_main

        root = Path(context.get("project_dir", ".")).resolve()
        run_dir = Path(context.get("ontology_run_dir", "")).resolve()
        master_dir = Path(context.get("master_ontology_dir", "")).resolve() if context.get("master_ontology_dir") else None

        config = load_config_env(root / "config" / "env.env")
        client = None
        if not context.get("local_only", False):
            client = init_deepseek_client(config, no_proxy=False)

        topic = context.get("topic", "")
        if master_dir and master_dir.is_dir():
            loader = OntologyContextLoader(master_dir)
            onto_blocks = loader.load_from_master_dir(master_dir, topic=topic)
        else:
            loader = OntologyContextLoader(str(run_dir))
            onto_blocks = loader.load()

        output_dir = Path(context.get("output_dir", "outputs/deepseek_tools/bom_analysis")).resolve()

        session = AnalysisSession(
            ontology_blocks=onto_blocks,
            background=context.get("background", "（未提供背景描述）"),
            question=context.get("question", "请进行分析"),
            ontology_run_dir=str(master_dir) if master_dir else str(run_dir),
            output_dir=output_dir,
            llm_client=client,
            llm_config={
                "model": config.get("DEEPSEEK_MODEL", "deepseek-chat"),
                "temperature": context.get("temperature", 0.15),
                "max_tokens": context.get("max_tokens", 8000),
            },
        )

        planner = PlannerAgent(session)
        analyst = AnalystAgent(session)
        verifier = VerifierAgent(session)
        relevance_gate = RelevanceGate()

        csv_files = context.get("csv_files", [])
        session.load_data_sources(csv_files, max_rows=int(context.get("max_csv_rows", 300)))

        result = session.run(
            max_rounds=int(context.get("max_rounds", 4)),
            converge_threshold=float(context.get("converge_threshold", 0.82)),
            verbose=False,
        )
        return {
            "status": result.status if hasattr(result, "status") else "COMPLETED",
            "final_findings": result.final_findings if hasattr(result, "final_findings") else result,
            "output_dir": str(output_dir),
        }
