# linecandidates

Identifying emission line candidates using John's template fits.


### Emission Line Numbers ###

There are 60 V6.2 objects in the V4.4 line list. Of these:

* 41 lie to the left of the 0th order x-cutoff
* 43 have cleanliness measures of <0.95 in both grisms
* 47 are marked `has_line` in the template fits
* 54 are marked as `peaked`
* 42 are marked as both `has_line` and 'peaked'

* * * 

Defining the criteria as
`has_line` & `peaked` & `clean < 0.95` & `x < cutoff`:

* Total of 216 objects for review
* 23 objects recovered from the line list


### Repo Contents ###

* `addedlines` - diagnostic plots for `Par302_estimates.fits`
* `noaddedlines` - diagnostic plots for `Par302_no_added_lines_estimates.fits`

