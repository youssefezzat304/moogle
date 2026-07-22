{
  pkgs,
  ...
}:

{
  packages = with pkgs; [
    git
  ];

  scripts = {
    pytest.exec = ''uv run pytest "$@"'';
  };

  enterShell = ''
    if [ ! -L "$DEVENV_ROOT/.venv" ]; then
        ln -s "$DEVENV_STATE/venv/" "$DEVENV_ROOT/.venv"
    fi
  '';

  languages.python = {
    enable = true;
    # version = "3.12";

    uv = {
      enable = true;
      sync = {
        enable = true;
        groups = [
          "test"
        ];
      };
    };

    libraries = [ pkgs.zlib ];
  };
}
