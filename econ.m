x = linspace(0, 24, 1441);

out_w = zeros(525601, 1);
out_s = out_w;

% special case: first day does not have data for hour 0

bound_w = 1:24;
out_w_curr = spline(bound_w, w_2(bound_w), x).';
nonzero = 1:1441;
out_w(nonzero) = out_w_curr;

bound_s = sol_bound(bound_w, s_2);
nonzero = bound_s(1)*60:bound_s(length(bound_s))*60;
out_s(nonzero+1) = spline(bound_s, [0 s_2(bound_s) 0], nonzero/60).';

%plot(x, out_w(1:1441));
%hold on
%plot(bound_w, w_2(bound_w));
%plot(x, out_s(1:1441));
%hold on
%plot(bound_w, s_2(bound_w));

bound_s = 0:24;

for day=2:365
    hour_start = 24*(day-1);
    bound_w = hour_start:hour_start+24;
    
    nonzero_w = bound_w(1)*60:bound_w(length(bound_w))*60;
    out_w(nonzero_w+1) = spline(0:24, w_2(bound_w), x).';

    if day==50
        %plot(x, out_w(nonzero_w+1));
        %hold on
        %plot(0:24, w_2(bound_w));
    end
    
    bound_s = sol_bound(bound_w, s_2);
    bound_s2 = rem(bound_s, 24);
    nonzero_s = bound_s(1)*60:bound_s(length(bound_s))*60;
    nonzero_s2 = bound_s2(1)*60:bound_s2(length(bound_s2))*60;
    out_s(nonzero_s+1) = spline(bound_s2, [0 s_2(bound_s) 0], nonzero_s2/60).';

    if day==50
        %plot(x, out_s(nonzero_w+1));
        %hold on
        %plot(0:24, s_2(bound_w));
    end
end

out_w_new = wind_power(out_w.');
out_s = max(out_s, 0);

T_w = table(out_w);
T_s = table(out_s);

temp=1200:1248;
plot(temp*60, w_2(temp));
hold on
temp=72000:74880;
plot(temp, out_w(temp));
hold on
plot(temp, out_w_new(temp)*13/3);
xlabel("time (minutes since start of year)");
ylabel("wind speed (m/s) or power output (13/3 MW)");
legend("actual wind speed", "interpolated wind speed", "interpolated power output");
title("Actual vs. Interpolated Wind Speed and Wind Power Output, Days 50 and 51");
%xlabel("time (minutes since start of year)"), ylabel("solar radiation (W/m^2)");
%legend("actual solar radiation", "interpolated solar radiation");
%title("Actual vs. Interpolated Solar Radiation, Days 50 and 51");

%writetable(T_w, 'windpower.txt');
%writetable(T_s, 'solarpower.txt');

% caps the bound of solar power to indices corresponding to positive
% irradiation, plus one more adjacent
% inputs: bound = range of hours interested it
%         s = array of solar irradiation at each hour
% output: range of values interested in
function out=sol_bound(bound, s)
    first = bound(1);
    while s(first) == 0
        first = first+1;
    end
    last = first;
    while s(last) ~= 0
        last = last+1;
    end
    out = first-1:last;
end

% complete: given first derivatives at endpoint
% not a knot: third derivative continuity
% complete cubic spline: plot(x, spline(bound, [0 w(bound) 0], x))
% not a knot: plot(x, spline(bound, w(bound), x))

% computes the power output the wind turbine generates, given velocity
% error in the supplemental information: denominator should have v_ci
% instead of v
function out = wind_power(list)
    v_ci = 3;
    v_s = 13;
    v_co = 25;
    P_ws = 3;
    out = zeros(length(list), 1);
    i = 1;
    for v=list
        P_w = 0;
        if v > v_ci & v <= v_s
            P_w = ((v^3 - v_ci^3)/(v_s^3 - v_ci^3)) * P_ws;
        end
        if v > v_s & v <= v_co
            P_w = P_ws;
        end
        out(i) = P_w;
        i = i+1;
    end
end