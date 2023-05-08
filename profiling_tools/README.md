# -*- mode: org -*-



* basic
#+begin_src bash
pyinstrument <python-script>
#+end_src

* show all
#+begin_src bash
pyinstrument --show-all <python-script>
#+end_src

#+RESULTS:

* to html (vue)
pyinstrument -o profile.html <python-script>

* options
Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit
  --load=FILENAME       instead of running a script, load a profile session
                        from a pyisession file
  --load-prev=IDENTIFIER
                        instead of running a script, load a previous profile
                        session as specified by an identifier
  -m MODULE_NAME        run library module as a script, like 'python -m
                        module'
  --from-path           (POSIX only) instead of the working directory, look
                        for scriptfile in the PATH environment variable
  -o OUTFILE, --outfile=OUTFILE
                        save to <outfile>
  -r RENDERER, --renderer=RENDERER
                        how the report should be rendered. One of: 'text',
                        'html', 'json', 'speedscope' or python import path to
                        a renderer class. Defaults to the appropriate format
                        for the extension if OUTFILE is given, otherwise,
                        defaults to 'text'.
  -p RENDER_OPTION, --render-option=RENDER_OPTION
                        options to pass to the renderer, in the format
                        'flag_name' or 'option_name=option_value'. For
                        example, to set the option 'time', pass '-p
                        time=percent_of_total'. To pass multiple options, use
                        the -p option multiple times. You can set processor
                        options using dot-syntax, like '-p
                        processor_options.filter_threshold=0'. option_value is
                        parsed as a JSON value or a string.
  -t, --timeline        render as a timeline - preserve ordering and don't
                        condense repeated calls
  --hide=EXPR           glob-style pattern matching the file paths whose
                        frames to hide. Defaults to hiding non-application
                        code
  --hide-regex=REGEX    regex matching the file paths whose frames to hide.
                        Useful if --hide doesn't give enough control.
  --show=EXPR           glob-style pattern matching the file paths whose
                        frames to show, regardless of --hide or --hide-regex.
                        For example, use --show '*/<library>/*' to show frames
                        within a library that would otherwise be hidden.
  --show-regex=REGEX    regex matching the file paths whose frames to always
                        show. Useful if --show doesn't give enough control.
  --show-all            show everything
  --unicode             (text renderer only) force unicode text output
  --no-unicode          (text renderer only) force ascii text output
  --color               (text renderer only) force ansi color text output
  --no-color            (text renderer only) force no color text output
  -i INTERVAL, --interval=INTERVAL
                        Minimum time, in seconds, between each stack sample.
                        Smaller values allow resolving shorter duration
                        function calls but conversely incur a greater runtime
                        and memory consumption overhead. For longer running
                        scripts, setting a larger interval can help control
                        the rate at which the memory required to store the
                        stack samples increases.

* References
github : https://github.com/joerick/pyinstrument
official user guide: https://pyinstrument.readthedocs.io/en/latest/guide.html
