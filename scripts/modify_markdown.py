# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click>=8.4.1",
#     "pydantic>=2.13.4",
#     "pydantic-ai>=1.107.0",
#     "rich>=15.0.0",
# ]
# ///

from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
import logging
import re

import click
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

INLINE_LATEX_PATTERN = re.compile(
    r"(?<![\$\\])\$(?!\$)(?P<content>(?:\\.|[^\$\\])+?)\$(?!\$)"
)

console = Console(stderr=True)
log = logging.getLogger("md_fixup")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_path=verbose,
                markup=True,
                omit_repeated_times=False,
            )
        ],
    )


@dataclass(frozen=True)
class InlineLatex:
    line_no: int
    column: int
    content: str


class Conversion(BaseModel):
    index: int = Field(description="The index of the input item being converted")
    latex: str = Field(description="The original LaTeX source")
    unicode: str = Field(description="The unicode-rendered equivalent")


class ConversionBatch(BaseModel):
    conversions: list[Conversion]


def create_file_reader(filepath: Path) -> Callable[[], str | None]:
    file_iter: Generator[str, None, None] = (
        line for line in filepath.open("r", encoding="utf-8")
    )

    def reader() -> str | None:
        return next(file_iter, None)

    return reader


def find_inline_latex(filepath: Path) -> list[InlineLatex]:
    reader = create_file_reader(filepath)
    results: list[InlineLatex] = []
    line_no = 1
    while (line := reader()) is not None:
        for m in INLINE_LATEX_PATTERN.finditer(line):
            il = InlineLatex(
                line_no=line_no,
                column=m.start(),
                content=m.group("content"),
            )
            log.debug(
                "match [dim]L%d:C%d[/dim] [cyan]%s[/cyan]",
                il.line_no,
                il.column,
                il.content,
            )
            results.append(il)
        line_no += 1
    return results


def batched[T](iterable: list[T], n: int) -> Iterator[list[T]]:
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


def build_agent(model: str, base_url: str) -> Agent[None, ConversionBatch]:
    ollama = OpenAIChatModel(
        model_name=model,
        provider=OpenAIProvider(base_url=base_url),
        settings=ModelSettings(max_tokens=256, temperature=0, thinking=False),
    )
    return Agent(
        ollama,
        output_type=ConversionBatch,
        system_prompt=(
            "You convert inline LaTeX math into the closest unicode-only "
            "equivalent. Use unicode sub/superscripts, Greek letters, and math "
            "symbols. Do not wrap output in LaTeX delimiters. Echo back the "
            "index of each input item exactly, and convert every item given."
        ),
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool):
    setup_logging(verbose)


@cli.command()
@click.argument("filepath", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model", default="qwen3.5:9b", help="Ollama model name. Default: 'qwen3.5:9b'"
)
@click.option(
    "--base-url",
    default="http://localhost:11434/v1",
    help="Ollama OpenAI-compatible endpoint. Default: 'http://localhost:11434/v1'",
)
@click.option(
    "--batch-size",
    default=10,
    type=int,
    help="Number of LaTeX matches per Ollama call. Default: 10",
)
def latex_to_unicode(filepath: Path, model: str, base_url: str, batch_size: int):
    log.info("scanning [bold]%s[/bold]", filepath)
    matches = find_inline_latex(filepath)

    if not matches:
        log.warning("no inline LaTeX found")
        return

    log.info(
        "found [bold green]%d[/bold green] inline LaTeX span(s), "
        "model=[magenta]%s[/magenta] batch_size=%d",
        len(matches),
        model,
        batch_size,
    )

    agent = build_agent(model, base_url)
    total_batches = (len(matches) + batch_size - 1) // batch_size
    converted = 0
    failed = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("converting", total=total_batches)
        for batch_no, chunk in enumerate(batched(matches, batch_size), 1):
            prompt = "\n".join(f"{i}: {m.content}" for i, m in enumerate(chunk))
            try:
                result = agent.run_sync(prompt)
            except Exception as exc:
                failed += len(chunk)
                log.error("batch %d/%d failed: %s", batch_no, total_batches, exc)
                progress.advance(task)
                continue

            got = {c.index: c for c in result.output.conversions}
            for i, m in enumerate(chunk):
                conv = got.get(i)
                if conv is None:
                    failed += 1
                    log.warning(
                        "[yellow]dropped[/yellow] L%d:C%d [cyan]%s[/cyan] "
                        "(no conversion returned)",
                        m.line_no,
                        m.column,
                        m.content,
                    )
                    continue
                converted += 1
                log.debug(
                    "L%d:C%d [cyan]%s[/cyan] -> [green]%s[/green]",
                    m.line_no,
                    m.column,
                    conv.latex,
                    conv.unicode,
                )
                console.print(
                    f"  ({batch_no + 1}.{i + 1}) (OLD) [yellow]{conv.latex}[/yellow] -> "
                    f"(NEW) [green]{conv.unicode}[/green]"
                )
            progress.advance(task)

    log.info(
        "done: [green]%d converted[/green], [red]%d failed[/red]",
        converted,
        failed,
    )


if __name__ == "__main__":
    cli()
