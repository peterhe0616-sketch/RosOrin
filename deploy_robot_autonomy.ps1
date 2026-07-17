param(
    [string]$SshHost = "ros-robot",
    [string]$Container = "MentorPi",
    [switch]$Start
)

$ErrorActionPreference = "Stop"
$Package = Join-Path $PSScriptRoot "robot\rosorin_autonomy"
if (-not (Test-Path (Join-Path $Package "package.xml"))) {
    throw "ROS 2 package not found: $Package"
}

$Archive = Join-Path ([IO.Path]::GetTempPath()) "rosorin_autonomy.tgz"
try {
    tar -czf $Archive -C $Package .
    if ($LASTEXITCODE -ne 0) { throw "Unable to create deployment archive" }

    ssh $SshHost "mkdir -p /home/pi/docker/tmp/rosorin_autonomy"
    if ($LASTEXITCODE -ne 0) { throw "SSH connection failed" }
    scp $Archive "${SshHost}:/home/pi/docker/tmp/rosorin_autonomy/package.tgz"
    if ($LASTEXITCODE -ne 0) { throw "Package upload failed" }

    ssh $SshHost "tar -xzf /home/pi/docker/tmp/rosorin_autonomy/package.tgz -C /home/pi/docker/tmp/rosorin_autonomy"
    if ($LASTEXITCODE -ne 0) { throw "Remote extraction failed" }

    $Build = "source /opt/ros/humble/setup.bash && mkdir -p /home/ubuntu/ros2_ws/src/rosorin_autonomy && cp -a /home/ubuntu/shared/rosorin_autonomy/. /home/ubuntu/ros2_ws/src/rosorin_autonomy/ && cd /home/ubuntu/ros2_ws && colcon build --packages-select rosorin_autonomy --symlink-install"
    ssh $SshHost "docker exec $Container bash -lc '$Build'"
    if ($LASTEXITCODE -ne 0) { throw "ROS 2 package build failed" }

    Write-Host "ROSOrin autonomy package deployed without reinstalling ROS 2."
    if ($Start) {
        $Run = "source /home/ubuntu/shared/.typerc >/dev/null && source /opt/ros/humble/setup.bash && source /home/ubuntu/third_party_ros2/third_party_ws/install/setup.bash && source /home/ubuntu/ros2_ws/install/setup.bash && ros2 launch rosorin_autonomy system.launch.py"
        ssh $SshHost "docker exec -d $Container bash -lc '$Run >/home/ubuntu/shared/rosorin_autonomy.log 2>&1'"
        if ($LASTEXITCODE -ne 0) { throw "Autonomy launch failed" }
        Write-Host "Autonomy stack started. Log: /home/pi/docker/tmp/rosorin_autonomy.log"
    }
}
finally {
    Remove-Item -LiteralPath $Archive -Force -ErrorAction SilentlyContinue
}
