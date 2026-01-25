"""
Xorq Vignettes System

Vignettes are complete, working examples that demonstrate specific xorq patterns
and techniques. They are more comprehensive than simple templates and show
real-world use cases.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import textwrap

@dataclass
class Vignette:
    """Represents a code vignette with metadata."""
    name: str
    filename: str
    title: str
    description: str
    tags: List[str]
    difficulty: str  # "beginner", "intermediate", "advanced"

    @property
    def file_path(self) -> Path:
        """Get the full path to the vignette file."""
        return Path(__file__).parent / self.filename

    def exists(self) -> bool:
        """Check if the vignette file exists."""
        return self.file_path.exists()

    def read_content(self) -> str:
        """Read the vignette file content."""
        if not self.exists():
            raise FileNotFoundError(f"Vignette file not found: {self.file_path}")
        return self.file_path.read_text()

    def get_header_comment(self) -> str:
        """Generate a header comment for the vignette."""
        return f'''"""
{self.title}
{"=" * len(self.title)}

{self.description}

Tags: {", ".join(self.tags)}
Difficulty: {self.difficulty}

This is an xorq vignette demonstrating {self.name.replace("_", " ")}.
"""

'''

# Registry of available vignettes
VIGNETTES = [
    Vignette(
        name="penguins_classification_intro",
        filename="penguins_classification_intro.py",
        title="Introduction to xorq: Penguin Species Classification",
        description=textwrap.dedent("""
            A beginner-friendly introduction to xorq through a complete ML pipeline for
            classifying penguin species. This vignette is perfect as your first xorq project!

            Learn the fundamentals:
            - Deferred execution: Building computation graphs without executing
            - Expression composition: Creating reusable, composable data pipelines
            - ML integration: Using scikit-learn models in deferred expressions
            - Caching strategy: Efficient storage with ParquetCache
            - Metrics & visualization: Computing ML metrics and ROC curves in deferred mode

            The extensive inline commentary explains every concept, making this ideal
            for newcomers to xorq. Uses only two features (bill measurements) to keep
            the focus on xorq patterns rather than complex ML.
        """).strip(),
        tags=["beginner", "ml", "classification", "tutorial", "penguins", "roc", "metrics"],
        difficulty="beginner"
    ),
    Vignette(
        name="baseball_breakout_expr_scalar",
        filename="baseball_breakout_expr_scalar.py",
        title="Baseball Player Breakout Prediction with ExprScalarUDF",
        description=textwrap.dedent("""
            Demonstrates advanced ML patterns in xorq using ExprScalarUDF for model training
            and prediction. This vignette shows:
            - Feature engineering with window functions and lag operations
            - Training a RandomForest model using pandas UDAF
            - Making predictions with ExprScalarUDF that accepts trained model as input
            - Working with baseball statistics to predict player breakout seasons

            The pattern shows how to create a fully deferred ML pipeline where the model
            is trained as an expression and then used for predictions without eager execution.
        """).strip(),
        tags=["ml", "udaf", "expr_scalar_udf", "feature_engineering", "sports_analytics"],
        difficulty="advanced"
    ),
    Vignette(
        name="ml_pipeline_roc_visualization",
        filename="ml_pipeline_roc_visualization.py",
        title="Complete ML Pipeline with ROC Visualization",
        description=textwrap.dedent("""
            A production-ready ML pipeline showcasing functional composition and advanced
            visualization patterns. This comprehensive vignette demonstrates:
            - Building a complete classifier pipeline using .pipe() for functional composition
            - Computing standard ML metrics (accuracy, precision, recall, F1, ROC-AUC)
            - Generating ROC curves and storing them as binary data in a UDAF
            - Using xorq.vendor.ibis for enhanced operators like .cache() and .into_backend()
            - Maintaining fully deferred execution until final execute() calls

            Perfect for understanding how to build production ML pipelines that leverage
            xorq's deferred execution model while producing publication-quality visualizations.
            The extensive inline commentary explains each pattern and design decision.
        """).strip(),
        tags=["ml", "visualization", "roc", "udaf", "pipeline", "functional_programming", "metrics"],
        difficulty="intermediate"
    ),
]

def list_vignettes() -> List[Vignette]:
    """Return list of all available vignettes."""
    return VIGNETTES

def get_vignette(name: str) -> Optional[Vignette]:
    """Get a specific vignette by name."""
    for vignette in VIGNETTES:
        if vignette.name == name:
            return vignette
    return None

def get_vignette_names() -> List[str]:
    """Get list of all vignette names."""
    return [v.name for v in VIGNETTES]

def scaffold_vignette(name: str, dest: Optional[Path] = None, overwrite: bool = False) -> Path:
    """
    Scaffold a vignette to a destination path.

    Args:
        name: Name of the vignette to scaffold
        dest: Destination path (defaults to vignettes/<name>.py)
        overwrite: Whether to overwrite existing file

    Returns:
        Path to the scaffolded file
    """
    vignette = get_vignette(name)
    if not vignette:
        raise ValueError(f"Vignette '{name}' not found")

    if dest is None:
        dest = Path(f"vignettes/{vignette.filename}")
    else:
        dest = Path(dest)

    # Create parent directory if needed
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {dest}. Use --overwrite to replace.")

    # Read the original content
    content = vignette.read_content()

    # Add header comment if not already present
    if not content.startswith('"""'):
        content = vignette.get_header_comment() + content

    # Write to destination
    dest.write_text(content)

    return dest

def format_vignette_list() -> str:
    """Format the list of vignettes for display."""
    lines = []
    lines.append("Available xorq vignettes:")
    lines.append("")

    # Group by difficulty
    by_difficulty = {}
    for v in VIGNETTES:
        if v.difficulty not in by_difficulty:
            by_difficulty[v.difficulty] = []
        by_difficulty[v.difficulty].append(v)

    # Display in order: beginner, intermediate, advanced
    for level in ["beginner", "intermediate", "advanced"]:
        if level in by_difficulty:
            lines.append(f"{level.upper()} LEVEL:")
            for v in by_difficulty[level]:
                lines.append(f"  • {v.name}")
                # Wrap description to 70 chars with proper indentation
                wrapped = textwrap.fill(v.description.split('\n')[0],
                                       width=70,
                                       initial_indent="    ",
                                       subsequent_indent="    ")
                lines.append(wrapped)
                lines.append(f"    Tags: {', '.join(v.tags)}")
                lines.append("")

    return "\n".join(lines)

def format_vignette_details(name: str) -> str:
    """Format detailed information about a specific vignette."""
    vignette = get_vignette(name)
    if not vignette:
        return f"Vignette '{name}' not found"

    lines = []
    lines.append(f"VIGNETTE: {vignette.name}")
    lines.append("=" * (10 + len(vignette.name)))
    lines.append("")
    lines.append(f"Title: {vignette.title}")
    lines.append(f"Difficulty: {vignette.difficulty}")
    lines.append(f"Tags: {', '.join(vignette.tags)}")
    lines.append("")
    lines.append("Description:")
    lines.append("-" * 12)
    lines.append(vignette.description)
    lines.append("")
    lines.append("To scaffold this vignette:")
    lines.append(f"  xorq agent vignette scaffold {vignette.name}")
    lines.append("")
    lines.append("Or with custom destination:")
    lines.append(f"  xorq agent vignette scaffold {vignette.name} --dest my_{vignette.name}.py")

    return "\n".join(lines)