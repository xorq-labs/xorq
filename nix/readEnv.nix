{ pkgs }:
let
  inherit (pkgs.lib.attrsets) nameValuePair;
  inherit (pkgs.lib.strings) splitString;
  readLines = file:
    let
      fileContents = builtins.readFile file;
      lines = builtins.filter
        (line: line != "")
        (splitString "\n" fileContents);
    in
      lines;
  readEnvFile = file:
    let
      parseLine = line:
        let
          pattern = "(export )?([^=]+)=(.*)";
          parts = builtins.match pattern line;
          name = builtins.elemAt parts 1;
          value = builtins.elemAt parts 2;
        in
          nameValuePair name value;
      lines = readLines file;
      attrs = builtins.listToAttrs (map parseLine lines);
    in
      attrs;
  get-env-default = default: name:
    let
      value = builtins.getEnv name;
    in
      if (builtins.stringLength value != 0) then value else default;
in
{
  inherit readEnvFile get-env-default;
}
