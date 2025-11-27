# Documentation linting setup

We use Vale to enforce our documentation style (Xorq + Google developer style guide).

> [!NOTE]
> The `.vale.ini` file and `styles/` directory are already in this repo. You only need to:
> * Install the Vale CLI.
> * (Optionally) hook it into your editor (VS Code).

## 1. Install the Vale CLI

You need the `vale` CLI installed and available on your system `PATH`.

### macOS (Homebrew)

```bash
brew install vale
```

### Ubuntu / Linux (Snap)

```bash
# --classic is required so Vale can access files on your system
sudo snap install vale --classic
```

### Windows (Chocolatey or Winget)

```bash
choco install vale
```

### Verify the installation

Close your terminal, open a new one, and run:

```bash
vale --version
```

If you see a version number, Vale is installed correctly.

> [!IMPORTANT]
> You do not need to run `vale sync` – the `styles/` folder is already committed in this repo.

## 2. VS Code integration (recommended)

If you want to see Vale suggestions while you edit the docs, you can use the VS Code extension.

### Step A: Install the extension

1. Open VS Code in this project (for example, run `code .` from the repo root).
2. Go to Extensions (`Ctrl+Shift+X` / `Cmd+Shift+X`).
3. Search for "Vale VSCode" (by Chris Chinchilla) and install it.

### Step B: Make sure the CLI is on `PATH`

Open a terminal inside VS Code and check that VS Code can see `vale`.

#### macOS / Linux (integrated terminal in VS Code)

```bash
vale --version
```

#### Windows (PowerShell in VS Code)

```bash
vale --version
```

> [!WARNING]
> If that prints a version, the extension can use it. If not, follow step C and restart VS Code

### Step C: Configure Vale in VS Code

To get live feedback as you edit the docs:

1. In VS Code, open Settings (`Ctrl+,` / `Cmd+,`).
2. In the search box, type "Vale CLI".
3. Under User (or Workspace if you want it only for this project):
   * In Vale CLI: Config, enter:

```
.vale.ini
```

   * Enable Vale CLI: Install Vale.
   * Set Vale CLI: Min Alert Level to `suggestion` from the dropdown.
4. Reload VS Code.

After this, when you open any `.qmd` or `.md` file under `docs/`, Vale will show suggestions inline (squiggly underlines) and in the Problems panel as you write.

## 3. Run Vale from the CLI

You can also run Vale manually from the repo root.

### Lint the entire docs tree

```bash
vale docs/
```

### Lint a specific file

```bash
vale docs/tutorials/getting_started/quickstart.qmd
```

> [!TIP]
> Use this as a final check before opening a PR or pushing documentation changes.