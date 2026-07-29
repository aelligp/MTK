using JLD2, CairoMakie, MathTeXEngine

CairoMakie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

# directory populated by SillConvection.jl (its `figdir`): `final_$(n)x$(n).jld2`, `snapshot_*_$(n)x$(n).jld2`
data_dir = "GMD_test_run_2026-07-29"
n = 256
yr = 3600 * 24 * 365

let
    fig = Figure(size = (1400, 700), fontsize = 20)

    # --- top row: temperature snapshots ---
    snapshot_files = filter(
        f -> occursin("_$(n)x$(n).jld2", f) && startswith(f, "snapshot_"),
        readdir(data_dir)
    )
    snapshot_times = [parse(Int, split(f, "_")[2]) for f in snapshot_files]
    order = sortperm(snapshot_times)
    snapshot_files = snapshot_files[order]

    local heat
    for (j, f) in enumerate(snapshot_files)
        d = load(joinpath(data_dir, f))
        t_snap = round(d["time"] / yr; digits = 1)
        ylabel = j == 1 ? L"$$z" : ""
        ax = Axis(
            fig[1, j], aspect = DataAspect(), title = L"$$t = %$(t_snap)\;\mathrm{yr}",
            xlabel = L"$$x", ylabel = ylabel,
            titlesize = 24, xlabelsize = 18, ylabelsize = 18,
            xticklabelsize = 16, yticklabelsize = 16,
        )
        heat = heatmap!(ax, d["xci"][1], d["xci"][2], d["T"], colormap = :lipari)

        panel_label = string("(", Char('a' + (j - 1)), ")")
        inset_ax = Axis(fig[1, j], width = Relative(0.37), height = Relative(0.12), halign = :left, valign = :top)
        hidedecorations!(inset_ax); hidespines!(inset_ax)
        text!(inset_ax, 0.1, 0.5, text = panel_label, space = :relative, align = (:center, :center), fontsize = 25, color = :black)
    end
    Colorbar(fig[1, length(snapshot_files) + 1], heat; label = L"$$T \;[^\circ\mathrm{C}]", labelsize = 20, ticklabelsize = 16)

    # --- bottom row: viscosity (left axis, black) and H2O dissolved/exsolved (right axis) vs time ---
    d_final = load(joinpath(data_dir, "final_$(n)x$(n).jld2"))
    t_evo   = d_final["time_vec"] ./ yr

    ax_visc = Axis(
        fig[2, 1:length(snapshot_files)], xlabel = L"$$Time \;[\mathrm{yr}]", ylabel = L"$$\eta \;[\mathrm{Pa\,s}]",
        yscale = log10,
        xlabelsize = 24, ylabelsize = 24, xticklabelsize = 14, yticklabelsize = 14,
    )
    panel_label = string("(", Char('a' + length(snapshot_files)), ")")
    inset_ax = Axis(fig[2, 1], width = Relative(0.37), height = Relative(0.12), halign = :left, valign = :top)
    hidedecorations!(inset_ax); hidespines!(inset_ax)
    text!(inset_ax, 0.15, 0.5, text = panel_label, space = :relative, align = (:center, :center), fontsize = 25, color = :black)

    lines!(ax_visc, t_evo, d_final["viscosity_evo"], color = :black, linewidth = 2, label = L"$$\eta")

    ax_water = Axis(
        fig[2, 1:length(snapshot_files)], ylabel = L"$$H_2O \;[\mathrm{wt\%}]",
        yaxisposition = :right, ylabelsize = 24, yticklabelsize = 14,
    )
    hidespines!(ax_water)
    hidexdecorations!(ax_water)
    linkxaxes!(ax_visc, ax_water)

    lines!(ax_water, t_evo, d_final["mH2O_diss_evo"] .* 100, color = :turquoise3, linewidth = 2, label = L"$$H_2O_{diss}")
    lines!(ax_water, t_evo, d_final["mH2O_exs_evo"] .* 100, color = :purple, linewidth = 2, label = L"$$H_2O_{exs}")

    axislegend(ax_visc, position = :lt, framevisible = true, labelsize = 16)
    axislegend(ax_water, position = :rt, framevisible = true, labelsize = 16)

    display(fig)
    out = joinpath(@__DIR__, "SillConvection_single_resolution_$(n)x$(n)")
    for ext in ("png", "pdf")
        save("$out.$ext", fig)
    end
    println("Saved $out.{png,pdf}")
end
