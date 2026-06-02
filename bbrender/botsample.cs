using System.Collections.Generic;
using Newtonsoft.Json;

public class CPHInline
{
    public bool Execute()
    {
        SendClear();
        CPH.Wait(500);

        var samples = new List<Sample>
        {
            // new Sample("Font 32px", "Absolute 32px text", 3500, null, null, null, 32),
            // new Sample("Font 48px", "Absolute 48px text", 3500, null, null, null, 48),
            // new Sample("Font 72px", "Absolute 72px text", 3500, null, null, null, 72),
            // new Sample("Font 104px", "Absolute 104px text", 3500, null, null, null, 104),

            // new Sample("Anchor Center", "center x=0 y=0", 3500, "center", 0, 0, 56),
            // new Sample("Anchor Top Left", "top-left x=80 y=60", 3500, "top-left", 80, 60, 48),
            // new Sample("Anchor Top", "top x=0 y=60", 3500, "top", 0, 60, 48),
            // new Sample("Anchor Top Right", "top-right x=80 y=60", 3500, "top-right", 80, 60, 48),
            // new Sample("Anchor Left", "left x=80 y=0", 3500, "left", 80, 0, 48),
            // new Sample("Anchor Right", "right x=80 y=0", 3500, "right", 80, 0, 48),
            // new Sample("Anchor Bottom Left", "bottom-left x=80 y=60", 3500, "bottom-left", 80, 60, 48),
            // new Sample("Anchor Bottom", "bottom x=0 y=60", 3500, "bottom", 0, 60, 48),
            // new Sample("Anchor Bottom Right", "bottom-right x=80 y=60", 3500, "bottom-right", 80, 60, 48),

            // new Sample("Center Nudge", "center nudged x=180 y=-120", 3500, "center", 180, -120, 48),
            // new Sample("Right Inset", "right edge 160px inward", 3500, "right", 160, 0, 48),
            // new Sample("Bottom Inset", "bottom edge 120px upward", 3500, "bottom", 0, 120, 48)

            // new Sample("Plain", "Plain websocket text renders safely."),
            // new Sample("Formatting", "Normal [b]Bold[/b] [i]Italic[/i] [u]Underline[/u] [s]Strike[/s]"),
            // new Sample("Reset", "[b][i]Styled[/all] plain"),
            // new Sample("Malformed", "[b]bold [i]italic[/b] trailing[/i]"),
            // new Sample("Colors", "[color=dodger-blue]Blue[/color] [bg=yellow][color=black]Highlight[/color][/bg] [opacity=0.45]Faded[/opacity]"),
            // new Sample("Gradient", "[gradient stops=\"red 0%,gold 50%,blue 100%\"]Gradient text[/gradient]"),
            // new Sample("Glow", "[outline color=black size=2][shadow color=gray x=3 y=3][glow color=gold size=8]Readable glow[/glow][/shadow][/outline]"),
            // new Sample("Wave Bounce", "[wave amp=24 freq=2][bounce amp=12 freq=3]Nested animation keeps spaces[/bounce][/wave]"),
            // new Sample("Rainbow", "[rainbow speed=2]Color cycling rainbow spaces[/rainbow]"),
            // new Sample("Blink", "[blink freq=1]Whole tag blink spaces[/blink]"),
            // new Sample("Flip", "[flip axis=x speed=0.6]Flip X spaces[/flip]  [flip axis=y speed=0.6]Flip Y spaces[/flip]"),
            // new Sample("Tornado", "[tornado radius=18 freq=0.75]Orbiting letters keep spaces[/tornado]"),
            // new Sample("Typewriter", "[typewriter speed=8 cursor=1]Typed reveal keeps spaces[/typewriter]", 6000),
            // new Sample("Hacker", "[hacker speed=8 loop=2 seed=demo]Decoded text keeps spaces[/hacker]", 6000),
            // new Sample("Fade", "[fade start=3 length=12]First letters solid, later letters fade[/fade]"),
            // new Sample("Slide", "[slide dir=left speed=80]Left[/slide] [slide dir=right speed=80]Right[/slide]"),
            // new Sample("Zoom", "[zoom from=0 to=1 speed=2]Zoom in[/zoom]"),
            // new Sample("Dropcap", "[dropcap]Signal starts with a large first letter and then wraps into normal body text over multiple lines.[/dropcap]", 6000),
            // new Sample("Wrap", "[wrap=char]SupercalifragilisticexpialidociousKeepsGoingWithoutSpaces[/wrap]", 6000),
            // new Sample("Positioned", "[wave]Bottom right absolute 72px[/wave]", 5000, "bottom-right", 80, 60, 72)
            new Sample("Zot", "[zoom from=0 to=1 speed=2][font=Nunito][color=#fd6b0a][size=60][b][stroke color=#f0e90c width=3]Your text with one [pulse freq=0.5]word[/pulse] pulsing slowly[/stroke][/b][/size][/color][/font]", 6000),
            new Sample(
                "Root Transition",
                "[font=Nunito][color=#fd6b0a][size=60][b][stroke color=#f0e90c width=3]Payload transition zooms in and fades out[/stroke][/b][/size][/color][/font]",
                5000,
                transitionIn: "zoom",
                transitionOut: "fade",
                inTime: 400,
                outTime: 700,
                ease: "out",
                scale: 0.08
            ),

        };

        foreach (var sample in samples)
        {
            Send(sample);
            CPH.Wait(sample.Duration + 700);
        }

        SendClear();
        return true;
    }

    private void Send(Sample sample)
    {
        var payload = new Dictionary<string, object>
        {
            { "type", "bbcode.render" },
            { "bbcode", "[small]" + sample.Name + "[/small][newline]" + sample.BBCode },
            { "duration", sample.Duration }
        };

        if (sample.Anchor != null) payload["anchor"] = sample.Anchor;
        if (sample.X.HasValue) payload["x"] = sample.X.Value;
        if (sample.Y.HasValue) payload["y"] = sample.Y.Value;
        if (sample.FontSize.HasValue) payload["fontSize"] = sample.FontSize.Value;
        if (sample.TransitionIn != null || sample.TransitionOut != null)
        {
            var transition = new Dictionary<string, object>();
            if (sample.TransitionIn != null) transition["in"] = sample.TransitionIn;
            if (sample.TransitionOut != null) transition["out"] = sample.TransitionOut;
            if (sample.InTime.HasValue) transition["inTime"] = sample.InTime.Value;
            if (sample.OutTime.HasValue) transition["outTime"] = sample.OutTime.Value;
            if (sample.Ease != null) transition["ease"] = sample.Ease;
            if (sample.Scale.HasValue) transition["scale"] = sample.Scale.Value;
            payload["transition"] = transition;
        }

        CPH.WebsocketBroadcastJson(JsonConvert.SerializeObject(payload));
    }

    private void SendClear()
    {
        var payload = new
        {
            type = "bbcode.render",
            command = "clear"
        };

        CPH.WebsocketBroadcastJson(JsonConvert.SerializeObject(payload));
    }

    private class Sample
    {
        public string Name;
        public string BBCode;
        public int Duration;
        public string Anchor;
        public int? X;
        public int? Y;
        public int? FontSize;
        public string TransitionIn;
        public string TransitionOut;
        public int? InTime;
        public int? OutTime;
        public string Ease;
        public double? Scale;

        public Sample(
            string name,
            string bbcode,
            int duration = 4000,
            string anchor = null,
            int? x = null,
            int? y = null,
            int? fontSize = null,
            string transitionIn = null,
            string transitionOut = null,
            int? inTime = null,
            int? outTime = null,
            string ease = null,
            double? scale = null
        )
        {
            Name = name;
            BBCode = bbcode;
            Duration = duration;
            Anchor = anchor;
            X = x;
            Y = y;
            FontSize = fontSize;
            TransitionIn = transitionIn;
            TransitionOut = transitionOut;
            InTime = inTime;
            OutTime = outTime;
            Ease = ease;
            Scale = scale;
        }
    }
}
