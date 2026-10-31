# Scientific workflow source snapshots

These files preserve the current numerical and inference tools in their original project-relative layouts. Consult the [software guide](../docs/software.md#research-workflow-sources) for the role of each directory and required runtime artifacts.

This source snapshot includes historical development and diagnostics as well as accepted interfaces. Presence of a script does not certify its method or authorize rerunning a closed study. Original runtime guards remain in place. Gigabyte-scale operator caches and full ensembles are not bundled; the compact validated row bank used by the public interface is in `../data/experimental/banks`.

The installable, locally demonstrated APIs are in `../src`. Research scripts keep their original imports and may require the original runtime capsule, compiled libraries, configurations, manifests, and data. Source origins and hashes are in `../SOURCE_MANIFEST.json`. Tiktaalik-derived code retains its upstream license and notices.
