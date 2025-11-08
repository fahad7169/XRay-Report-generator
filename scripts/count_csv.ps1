$files = @(
    'Final_Train_Data.csv',
    'Final_CV_Data.csv',
    'Final_Test_Data.csv',
    'train.csv',
    'test.csv',
    'cv.csv',
    'processed_train.csv',
    'processed_cv.csv',
    'processed_test.csv'
)

foreach ($f in $files) {
    if (Test-Path $f) {
        $lines = (Get-Content -Path $f | Measure-Object -Line).Lines
        $header = Get-Content -Path $f -First 1
        $cols = ($header -split ',').Length
        $rows = [math]::Max($lines - 1, 0)
        Write-Output ("{0}`tRows:{1}`tCols:{2}`tLines:{3}" -f $f, $rows, $cols, $lines)
    } else {
        Write-Output ("{0}`tMISSING" -f $f)
    }
}



