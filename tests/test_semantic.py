from __future__ import annotations

from prscope.semantic import (
    CODE_EXTENSIONS,
    SKIP_DIRS,
    CodeChunk,
    extract_keywords_from_pr,
    extract_matching_files,
    read_local_files,
    search_code_by_keywords,
)


def test_code_chunk_summary():
    chunk = CodeChunk(
        path="src/auth/handler.py",
        content="def authenticate(user):\n    return True",
        start_line=1,
        end_line=2,
    )
    summary = chunk.summary()
    assert "src/auth/handler.py" in summary
    assert "def authenticate" in summary


def test_skip_dirs():
    assert ".git" in SKIP_DIRS
    assert "node_modules" in SKIP_DIRS
    assert "__pycache__" in SKIP_DIRS


def test_code_extensions():
    assert ".py" in CODE_EXTENSIONS
    assert ".ts" in CODE_EXTENSIONS
    assert ".tsx" in CODE_EXTENSIONS


def test_read_local_files(tmp_path):
    # Create test files
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print('hello')")
    (src / "utils.ts").write_text("export const foo = 1;")

    # Create ignored directory
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "pkg.js").write_text("ignored")

    chunks = read_local_files(tmp_path)

    paths = [c.path for c in chunks]
    assert "src/main.py" in paths
    assert "src/utils.ts" in paths
    # node_modules should be skipped
    assert not any("node_modules" in p for p in paths)


def test_extract_matching_files(tmp_path):
    # Create test files
    auth = tmp_path / "src" / "auth"
    auth.mkdir(parents=True)
    (auth / "login.py").write_text("def login(): pass")
    (auth / "logout.py").write_text("def logout(): pass")

    api = tmp_path / "src" / "api"
    api.mkdir(parents=True)
    (api / "users.py").write_text("def get_users(): pass")

    # Match by feature paths
    matched = extract_matching_files(
        repo_root=tmp_path,
        pr_files=["something/login.py"],  # Should match by filename
        feature_paths=["**/auth/**"],
    )

    paths = [c.path for c in matched]
    assert any("login.py" in p for p in paths)


def test_extract_keywords_from_pr():
    # Test security keywords
    title = "fix(security): restrict local path extraction to prevent LFI"
    body = "This PR prevents path traversal attacks by sanitize user input"

    keywords = extract_keywords_from_pr(title, body)

    assert "security" in keywords
    assert "path traversal" in keywords
    assert "lfi" in keywords
    assert "restrict" in keywords


def test_extract_keywords_from_pr_finds_function_names():
    title = "Add isPathSafe validation"
    body = "Implements validatePath and sanitizePath functions"

    keywords = extract_keywords_from_pr(title, body)

    # Should find camelCase function names
    assert any("isPathSafe" in k for k in keywords)
    assert any("validate" in k.lower() for k in keywords)
    assert any("sanitize" in k.lower() for k in keywords)


def test_search_code_by_keywords(tmp_path):
    # Create test files with security-related content
    security = tmp_path / "lib" / "security"
    security.mkdir(parents=True)
    (security / "index.ts").write_text("""
        // Path traversal prevention
        export function isPathSafe(path: string): boolean {
            if (path.includes('..')) return false;
            return true;
        }

        export function sanitizePath(path: string): string {
            return path.replace(/\\.\\./g, '');
        }
    """)

    # Create unrelated file
    utils = tmp_path / "lib" / "utils"
    utils.mkdir(parents=True)
    (utils / "helpers.ts").write_text("export const add = (a, b) => a + b;")

    # Search for security keywords
    keywords = ["path traversal", "sanitize", "isPathSafe"]
    matches = search_code_by_keywords(tmp_path, keywords)

    # Should find the security file
    assert len(matches) >= 1
    assert any("security" in m.path for m in matches)
    # Unrelated file should not match (or be lower priority)
    if len(matches) > 1:
        assert "security" in matches[0].path  # Security file should be first


def test_extract_matching_files_with_keywords(tmp_path):
    # Create files with different paths but same concept
    # Simulates: PR fixes media/parse.ts, but local fix is in security/index.ts

    media = tmp_path / "media"
    media.mkdir()
    (media / "parse.ts").write_text("// unrelated media parsing")

    security = tmp_path / "security"
    security.mkdir()
    (security / "index.ts").write_text("""
        // Path security - prevents LFI and path traversal
        export function isPathSafe(path: string): boolean {
            if (path.includes('..')) return false;
            return true;
        }
    """)

    # PR is about LFI prevention in media/parse.ts
    # But keyword search should also find security/index.ts
    matched = extract_matching_files(
        repo_root=tmp_path,
        pr_files=["src/media/parse.ts"],  # Different path
        feature_paths=[],
        pr_title="fix(security): prevent LFI in media parser",
        pr_body="Restrict path traversal to prevent local file inclusion",
    )

    paths = [c.path for c in matched]
    # Should find security file via keyword search
    assert any("security" in p for p in paths)
