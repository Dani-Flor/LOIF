function posix_time = get_current_posix_time()
    % GET_CURRENT_POSIX_TIME Returns the current POSIX time (seconds since 1970-01-01 00:00:00 UTC)

    % Check if running in Octave
    if exist('OCTAVE_VERSION', 'builtin')
        % In Octave, use the 'time' function which directly returns POSIX time
        posix_time = time();
    else
        % In MATLAB, use the datetime and posixtime functions
        % It's recommended to specify "UTC" time zone to avoid ambiguity
        current_utc_datetime = datetime('now', 'TimeZone', 'UTC');
        posix_time = posixtime(current_utc_datetime);
    end
end

